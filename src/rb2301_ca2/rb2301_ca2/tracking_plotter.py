"""Laptop viewer: receive tracking_debug.py telemetry via ROS for live maps, graphs and CSV logs.

Run separately; enable --debug-tracking on path_planning. Display: 5 Hz, 50-second window.
All Matplotlib work stays on the main thread of this separate process.
"""
import argparse
from collections import deque
import csv
from datetime import datetime
from pathlib import Path
import json
import math
import time

import numpy as np

SOURCE = {'NONE': 0, 'SEGMENT': 1, 'CARROT': 2, 'ENDPOINT': 3}  # Numeric levels for reference plots.
MODE = {'ALIGN': 1, 'DRIVE': 2}  # All other states are shown as idle.
LABELS = ['Heading error (deg)', 'Lateral error (m)',
          'Forward command (m/s)', 'Turn command (rad/s)',
          'Heading reference', 'Motion mode']

CSV_FIELDS = ['run_id', 'seq', 'stamp', 'heading', 'lateral', 'forward',
              'turn', 'source', 'mode', 'goal', 'segment']


class TrackingPlot:
    """Display only: no PID, path planning, or movement commands."""
    def __init__(self, window=50.0):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        self.window = window
        self.samples = deque(maxlen=10000)  # Also bounded if source time freezes.
        self.run_id = None
        self.start_stamp = None  # Controller timestamp used as elapsed time zero.
        self.last_received = None  # Local receipt time for detecting a paused stream.
        self.scene = None
        self.scene_dirty = False  # Redraw static map/route only after a scene update.
        self.fig = plt.figure(figsize=(14, 10))
        layout = self.fig.add_gridspec(6, 2, width_ratios=[1, 1.4])  # Map left; six graphs right.
        self.map_ax = self.fig.add_subplot(layout[:, 0])  # Map spans all six rows.
        self.axes = []
        self.lines = []
        for i, label in enumerate(LABELS):
            ax = self.fig.add_subplot(layout[i, 1],
                                      sharex=self.axes[0] if self.axes else None)
            line, = ax.plot([], [], drawstyle='steps-post' if i >= 4 else 'default')  # Modes/references are discrete states.
            ax.set_title(label, loc='left', fontsize=10, pad=4)
            ax.grid(True, alpha=0.25)
            if i < 4:
                ax.axhline(0, color='black', linewidth=0.6)
            if i < 5:
                ax.tick_params(labelbottom=False)
            self.axes.append(ax)
            self.lines.append(line)
        self.axes[4].set_yticks(list(SOURCE.values()))
        self.axes[4].set_yticklabels(list(SOURCE), fontsize=8)
        self.axes[4].set_ylim(-0.3, 3.3)
        self.axes[5].set_yticks([0, 1, 2])
        self.axes[5].set_yticklabels(['IDLE', 'ALIGN', 'DRIVE'])
        self.axes[5].set_ylim(-0.3, 2.3)
        self.axes[5].set_xlabel('Elapsed controller time (s)')

        # Array axes are [x, y]. set_scene() transposes to image [row=y, col=x].
        self.map_image = self.map_ax.imshow(np.zeros((1, 1)), origin='lower',
            cmap=ListedColormap(['white', '#375a9e']), vmin=0, vmax=1,
            interpolation='nearest', extent=(0, 1, 0, 1))
        self.route_line, = self.map_ax.plot([], [], '-', color='tab:green', label='A* route')
        self.waypoint_line, = self.map_ax.plot([], [], 'rs', ms=4, label='Waypoints')
        self.goal_line, = self.map_ax.plot([], [], '*', color='goldenrod', ms=10, label='Goals')
        self.trail_line, = self.map_ax.plot([], [], '-', color='0.6', lw=1, label='Recent motion')
        self.robot_line, = self.map_ax.plot([], [], 'ko', ms=6, label='Robot')
        self.carrot_line, = self.map_ax.plot([], [], 'o', color='tab:orange', ms=7, label='Carrot')
        self.yaw_arrow = self.map_ax.annotate('', xy=(0, 0), xytext=(0, 0),
                                             arrowprops=dict(arrowstyle='->', color='black', lw=2))
        self.aim_arrow = self.map_ax.annotate('', xy=(0, 0), xytext=(0, 0),
                                             arrowprops=dict(arrowstyle='->', color='tab:orange', lw=2))
        self.map_ax.set(xlabel='x (m)', ylabel='y (m)', title='Waiting for map')
        self.map_ax.set_aspect('equal')
        self.map_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
                           ncol=2, fontsize=8)
        self.title = self.fig.suptitle('Waiting for tracking data')
        self.fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08,
                                 top=0.91, wspace=0.35, hspace=0.65)

    @staticmethod
    def set_points(line, points):
        xy = np.asarray(points, dtype=float).reshape(-1, 2)  # Accept coordinate pairs, including an empty list.
        line.set_data(xy[:, 0], xy[:, 1])

    def set_scene(self, scene):
        self.scene = scene
        self.scene_dirty = True  # Cache new scene; draw it on the next GUI refresh.

    def add_sample(self, sample):
        if (sample['run_id'] != self.run_id or
                (self.samples and sample['stamp'] < self.samples[-1]['stamp'])):
            self.samples.clear()  # New run or clock reset: discard the previous live window.
            self.run_id = sample['run_id']
            self.start_stamp = sample['stamp']
        self.samples.append(sample)  # Keep received samples, not just screen-refresh samples.
        while len(self.samples) > 1 and sample['stamp'] - self.samples[0]['stamp'] > self.window:
            self.samples.popleft()  # Remove old samples from display; the CSV keeps them.
        self.last_received = time.monotonic()  # Independent of the robot/simulation clock.

    def draw_scene(self):
        scene = self.scene
        grid = np.asarray(scene['grid'])
        x, y = scene['origin']
        res = scene['resolution']
        extent = (x, x + grid.shape[0]*res, y, y + grid.shape[1]*res)
        self.map_image.set_data((grid != 0).T)  # Convert [x, y] grid to image rows/columns.
        self.map_image.set_extent(extent)
        self.map_ax.set_xlim(extent[:2])
        self.map_ax.set_ylim(extent[2:])
        for key, line in [('path', self.route_line), ('waypoints', self.waypoint_line)]:
            cells = np.asarray(scene[key], dtype=float).reshape(-1, 2)
            self.set_points(line, np.array([x, y]) + (cells + 0.5)*res)  # Plot grid-cell centres in metres.
        self.set_points(self.goal_line, scene['goals'])
        self.map_ax.set_title('Simulation map' if scene['simulation'] else 'Real maze (maze frame)')
        self.scene_dirty = False

    def redraw(self):
        if self.scene_dirty:  # Update the static map only when its scene changes.
            self.draw_scene()
        if not self.samples:
            self.fig.canvas.draw_idle()
            return
        rows = list(self.samples)
        latest = rows[-1]
        # Insert gaps so missing packets do not look like measured convergence.
        times, values = [], [[] for _ in range(6)]
        previous = None
        for row in rows:
            t = row['stamp'] - self.start_stamp
            if previous and (row['seq'] != previous['seq'] + 1 or
                             row['stamp'] - previous['stamp'] > 0.25):
                times.append(t)
                for series in values:
                    series.append(np.nan)  # Break the line across missing samples.
            times.append(t)
            data = [row['heading'], row['lateral'], row['forward'], row['turn'],
                    SOURCE[row['source']], MODE.get(row['mode'], 0)]  # Map states to labelled y-axis levels.
            for series, value in zip(values, data):
                series.append(np.nan if value is None else value)
            previous = row
        for line, series in zip(self.lines, values):
            line.set_data(times, series)
        for i in [0, 1, 2, 3]:
            self.axes[i].relim()
            self.axes[i].autoscale_view(scalex=False)  # Fit signal amplitude; keep a shared time window.
        end = max(self.window, times[-1])
        self.axes[0].set_xlim(max(0, end-self.window), end)

        same_scene = self.scene is not None and self.scene['run_id'] == self.run_id  # Avoid mixing runs on the map.
        pose = latest['pose'] if same_scene else None
        self.set_points(self.trail_line, [r['pose'][:2] for r in rows
                                         if r['pose'] is not None] if same_scene else [])
        self.set_points(self.robot_line, [pose[:2]] if pose else [])
        target = latest['target'] if same_scene else None
        self.set_points(self.carrot_line, [target] if target else [])
        self.yaw_arrow.set_visible(pose is not None)
        self.aim_arrow.set_visible(pose is not None and latest['heading'] is not None)
        if pose:
            x, y, heading = pose
            for arrow, angle in [(self.yaw_arrow, heading),
                                 (self.aim_arrow, heading + (latest['heading'] or 0))]:  # Aim = robot yaw + heading error.
                angle = math.radians(angle)
                arrow.set_position((x, y))
                arrow.xy = (x + 0.25*math.cos(angle), y + 0.25*math.sin(angle))
        age = time.monotonic() - self.last_received
        status = ' | NO NEW DATA {:.1f}s'.format(age) if age > 1 else ''  # Flag telemetry silence, not a controller fault.
        self.title.set_text('{} | goal {}/{} | segment {}/{} | heading reference: {}{}'.format(
            latest['mode'], latest['goal'], latest['goals'], latest['segment'],
            latest['segments'], latest['source'], status))
        self.fig.canvas.draw_idle()


def save_history(csv_path, show=False):
    """Render all recorded samples, including those outside the live window."""
    import matplotlib.pyplot as plt

    with open(csv_path, newline='') as file:
        rows = list(csv.DictReader(file))  # Reload all received samples, beyond the live window.
    if not rows:
        print('No samples received; no history PNG generated.')
        return
    times, values = [], [[] for _ in range(6)]
    previous = None
    start = float(rows[0]['stamp'])
    offset = 0.0
    for row in rows:
        stamp = float(row['stamp'])
        restarted = previous and (row['run_id'] != previous['run_id'] or
                                   stamp < float(previous['stamp']))
        if restarted:
            offset = times[-1] + 0.05  # Place restarted runs after the preceding history.
            start = stamp
        t = offset + stamp - start
        if previous and (restarted or int(row['seq']) != int(previous['seq']) + 1 or
                         stamp - float(previous['stamp']) > 0.25):
            times.append(t)
            for series in values:
                series.append(np.nan)
        times.append(t)
        data = [float(row[k]) if row[k] else np.nan
                for k in ['heading', 'lateral', 'forward', 'turn']]
        data += [SOURCE[row['source']], MODE.get(row['mode'], 0)]
        for series, value in zip(values, data):
            series.append(value)  # Heading/lateral errors and commanded velocities, not measured speeds.
        previous = row
    fig, axes = plt.subplots(6, 1, figsize=(13, 11), sharex=True)
    for i, (ax, label, series) in enumerate(zip(axes, LABELS, values)):
        ax.plot(times, series, drawstyle='steps-post' if i >= 4 else 'default')
        ax.set_title(label, loc='left', fontsize=10)
        ax.grid(True, alpha=0.25)
        if i < 4:
            ax.axhline(0, color='black', lw=0.6)
    axes[4].set_yticks(list(SOURCE.values()))
    axes[4].set_yticklabels(list(SOURCE))
    axes[5].set_yticks([0, 1, 2])
    axes[5].set_yticklabels(['IDLE', 'ALIGN', 'DRIVE'])
    axes[-1].set_xlabel('Elapsed controller time (s; restarts separated by gaps)')
    fig.suptitle('Full received tracking history — ' + Path(csv_path).name)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output = Path(csv_path).with_suffix('.png')
    fig.savefig(output, dpi=150)  # Save a full-history PNG beside the CSV.
    print('Saved history: {}'.format(output.resolve()))
    if show:
        plt.show()
    plt.close(fig)


def main(args=None):
    # ROS and GUI dependencies are used only when launching this viewer.
    import matplotlib.pyplot as plt
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from rb2301_ca2.tracking_debug import sample_qos, scene_qos

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window', type=float, default=50.0, help='Visible seconds')
    parser.add_argument('--refresh', type=float, default=5.0, help='Screen refresh Hz')
    parser.add_argument('--csv', help='CSV output path (default: timestamped file in tracking_logs/)')
    parser.add_argument('--history', help='Open a saved CSV as a full-history graph')
    options, ros_args = parser.parse_known_args(args)
    if options.history:  # Replay an existing CSV instead of subscribing to live topics.
        save_history(options.history, show=True)
        return
    if options.window <= 0 or options.refresh <= 0:
        parser.error('--window and --refresh must be positive')
    rclpy.init(args=ros_args)
    node = Node('tracking_plotter')
    plot = TrackingPlot(options.window)
    csv_path = Path(options.csv) if options.csv else Path('tracking_logs') / (
        datetime.now().strftime('tracking_%Y%m%d_%H%M%S_%f') + '.csv')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    logfile = open(csv_path, 'w', newline='')
    print('Recording samples: {}'.format(csv_path.resolve()))
    writer = csv.DictWriter(logfile, CSV_FIELDS, extrasaction='ignore') if logfile else None  # Save selected telemetry fields.
    if writer:
        writer.writeheader()

    def receive_sample(msg):
        try:
            sample = json.loads(msg.data)  # Decode the JSON payload from tracking_debug.py.
            plot.add_sample(sample)  # Buffer for graphs, robot pose and recent trail.
            if writer:
                writer.writerow(sample)  # Record every valid received sample, independently of redraws.
        except (ValueError, KeyError, TypeError) as exc:
            node.get_logger().error('Invalid tracking sample: {}'.format(exc))

    def receive_scene(msg):
        try:
            plot.set_scene(json.loads(msg.data))  # Cache map, goals, planned route and waypoints.
        except (ValueError, KeyError, TypeError) as exc:
            node.get_logger().error('Invalid tracking map: {}'.format(exc))

    node.create_subscription(String, 'tracking/sample', receive_sample, sample_qos())  # Changing pose/errors/commands.
    node.create_subscription(String, 'tracking/scene', receive_scene, scene_qos())  # Retained map/route for late viewers.

    def poll_ros():
        if not rclpy.ok():
            plt.close(plot.fig)
            return
        # Drain queued callbacks frequently; limit work to keep GUI responsive.
        deadline = time.monotonic() + 0.005  # Spend at most about 5 ms processing ROS callbacks.
        for _ in range(100):
            rclpy.spin_once(node, timeout_sec=0.0)  # Process a callback without waiting for a message.
            if time.monotonic() >= deadline:
                break

    poll_timer = plot.fig.canvas.new_timer(interval=10)  # Poll ROS every 10 ms on the GUI thread.
    poll_timer.add_callback(poll_ros)
    redraw_timer = plot.fig.canvas.new_timer(interval=max(1, int(1000/options.refresh)))  # Redraw at 5 Hz by default.
    def refresh():
        plot.redraw()
        logfile.flush()  # Flush buffered CSV data regularly during the run.

    redraw_timer.add_callback(refresh)
    poll_timer.start()
    redraw_timer.start()
    node.get_logger().info('Viewer ready. Start path_planning with --debug-tracking.')
    try:
        plt.show()  # GUI event loop also drives the ROS-polling and redraw timers.
    except KeyboardInterrupt:
        pass
    finally:
        poll_timer.stop()
        redraw_timer.stop()
        if logfile:
            logfile.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        save_history(csv_path)  # On normal exit, convert the CSV to a full-history graph.


if __name__ == '__main__':
    main()
