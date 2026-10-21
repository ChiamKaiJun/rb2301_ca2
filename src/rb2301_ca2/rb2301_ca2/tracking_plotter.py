"""Laptop-only live map and graphs. Run separately from path_planning.

Receive every available control sample, redraw at 5 Hz, retain 20 seconds.
All Matplotlib work stays on the main thread of this separate process.
"""
import argparse
from collections import deque
import csv
import json
import math
import time

import numpy as np

SOURCE = {'NONE': 0, 'SEGMENT': 1, 'CARROT': 2, 'ENDPOINT': 3}
MODE = {'ALIGN': 1, 'DRIVE': 2}  # All other states are shown as idle.
CSV_FIELDS = ['run_id', 'seq', 'stamp', 'heading', 'lateral', 'forward',
              'turn', 'source', 'mode', 'goal', 'segment']


class TrackingPlot:
    """Display only: no PID, path planning, or movement commands."""
    def __init__(self, window=20.0):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        self.window = window
        self.samples = deque(maxlen=10000)  # Also bounded if source time freezes.
        self.run_id = None
        self.start_stamp = None
        self.last_received = None
        self.scene = None
        self.scene_dirty = False
        self.fig = plt.figure(figsize=(14, 10))
        layout = self.fig.add_gridspec(7, 2, width_ratios=[1, 1.4])
        self.map_ax = self.fig.add_subplot(layout[:, 0])
        self.axes = []
        labels = ['Heading error (deg)', 'Lateral error (m)',
                  'Forward cmd (m/s)', 'Turn cmd (rad/s)',
                  'Heading source', 'Mode', 'Goal / segment']
        self.lines = []
        for i, label in enumerate(labels):
            ax = self.fig.add_subplot(layout[i, 1],
                                      sharex=self.axes[0] if self.axes else None)
            line, = ax.plot([], [], drawstyle='steps-post' if i >= 4 else 'default')
            ax.set_ylabel(label, fontsize=9)
            ax.grid(True, alpha=0.25)
            if i < 4:
                ax.axhline(0, color='black', linewidth=0.6)
            if i < 6:
                ax.tick_params(labelbottom=False)
            self.axes.append(ax)
            self.lines.append(line)
        self.axes[4].set_yticks(list(SOURCE.values()))
        self.axes[4].set_yticklabels(list(SOURCE), fontsize=8)
        self.axes[4].set_ylim(-0.3, 3.3)
        self.axes[5].set_yticks([0, 1, 2])
        self.axes[5].set_yticklabels(['IDLE', 'ALIGN', 'DRIVE'])
        self.axes[5].set_ylim(-0.3, 2.3)
        self.lines[6].set_label('Goal')
        self.segment_line, = self.axes[6].plot([], [], drawstyle='steps-post',
                                              label='Segment', color='tab:orange')
        self.axes[6].legend(loc='upper left', fontsize=8, ncol=2)
        self.axes[6].set_xlabel('Controller time since first received sample (s)')

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
                                 top=0.91, wspace=0.42, hspace=0.25)

    @staticmethod
    def set_points(line, points):
        xy = np.asarray(points, dtype=float).reshape(-1, 2)
        line.set_data(xy[:, 0], xy[:, 1])

    def set_scene(self, scene):
        self.scene = scene
        self.scene_dirty = True

    def add_sample(self, sample):
        if (sample['run_id'] != self.run_id or
                (self.samples and sample['stamp'] < self.samples[-1]['stamp'])):
            self.samples.clear()
            self.run_id = sample['run_id']
            self.start_stamp = sample['stamp']
        self.samples.append(sample)
        while len(self.samples) > 1 and sample['stamp'] - self.samples[0]['stamp'] > self.window:
            self.samples.popleft()
        self.last_received = time.monotonic()

    def draw_scene(self):
        scene = self.scene
        grid = np.asarray(scene['grid'])
        x, y = scene['origin']
        res = scene['resolution']
        extent = (x, x + grid.shape[0]*res, y, y + grid.shape[1]*res)
        self.map_image.set_data((grid != 0).T)
        self.map_image.set_extent(extent)
        self.map_ax.set_xlim(extent[:2])
        self.map_ax.set_ylim(extent[2:])
        for key, line in [('path', self.route_line), ('waypoints', self.waypoint_line)]:
            cells = np.asarray(scene[key], dtype=float).reshape(-1, 2)
            self.set_points(line, np.array([x, y]) + (cells + 0.5)*res)
        self.set_points(self.goal_line, scene['goals'])
        self.map_ax.set_title('Simulation map' if scene['simulation'] else 'Real maze (maze frame)')
        self.scene_dirty = False

    def redraw(self):
        if self.scene_dirty:
            self.draw_scene()
        if not self.samples:
            self.fig.canvas.draw_idle()
            return
        rows = list(self.samples)
        latest = rows[-1]
        # Insert gaps so missing packets do not look like measured convergence.
        times, values = [], [[] for _ in range(8)]
        previous = None
        for row in rows:
            t = row['stamp'] - self.start_stamp
            if previous and (row['seq'] != previous['seq'] + 1 or
                             row['stamp'] - previous['stamp'] > 0.25):
                times.append(t)
                for series in values:
                    series.append(np.nan)
            times.append(t)
            data = [row['heading'], row['lateral'], row['forward'], row['turn'],
                    SOURCE[row['source']], MODE.get(row['mode'], 0), row['goal'], row['segment']]
            for series, value in zip(values, data):
                series.append(np.nan if value is None else value)
            previous = row
        for line, series in zip(self.lines, values):
            line.set_data(times, series)
        self.segment_line.set_data(times, values[7])
        for i in [0, 1, 2, 3, 6]:
            self.axes[i].relim()
            self.axes[i].autoscale_view(scalex=False)
        end = max(1.0, times[-1])
        self.axes[0].set_xlim(max(0, end-self.window), end)

        same_scene = self.scene is not None and self.scene['run_id'] == self.run_id
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
                                 (self.aim_arrow, heading + (latest['heading'] or 0))]:
                angle = math.radians(angle)
                arrow.set_position((x, y))
                arrow.xy = (x + 0.25*math.cos(angle), y + 0.25*math.sin(angle))
        age = time.monotonic() - self.last_received
        status = ' | NO NEW DATA {:.1f}s'.format(age) if age > 1 else ''
        self.title.set_text('{} | goal {}/{} | segment {}/{} | aiming: {}{}'.format(
            latest['mode'], latest['goal'], latest['goals'], latest['segment'],
            latest['segments'], latest['source'], status))
        self.fig.canvas.draw_idle()


def main(args=None):
    # ROS and GUI dependencies are used only when launching this viewer.
    import matplotlib.pyplot as plt
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from rb2301_ca2.tracking_debug import sample_qos, scene_qos

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window', type=float, default=20.0, help='Visible seconds')
    parser.add_argument('--refresh', type=float, default=5.0, help='Screen refresh Hz')
    parser.add_argument('--csv', help='Optional CSV file for all received samples')
    options, ros_args = parser.parse_known_args(args)
    if options.window <= 0 or options.refresh <= 0:
        parser.error('--window and --refresh must be positive')
    rclpy.init(args=ros_args)
    node = Node('tracking_plotter')
    plot = TrackingPlot(options.window)
    logfile = open(options.csv, 'w', newline='') if options.csv else None
    writer = csv.DictWriter(logfile, CSV_FIELDS, extrasaction='ignore') if logfile else None
    if writer:
        writer.writeheader()

    def receive_sample(msg):
        try:
            sample = json.loads(msg.data)
            plot.add_sample(sample)
            if writer:
                writer.writerow(sample)
        except (ValueError, KeyError, TypeError) as exc:
            node.get_logger().error('Invalid tracking sample: {}'.format(exc))

    def receive_scene(msg):
        try:
            plot.set_scene(json.loads(msg.data))
        except (ValueError, KeyError, TypeError) as exc:
            node.get_logger().error('Invalid tracking map: {}'.format(exc))

    node.create_subscription(String, 'tracking/sample', receive_sample, sample_qos())
    node.create_subscription(String, 'tracking/scene', receive_scene, scene_qos())

    def poll_ros():
        if not rclpy.ok():
            plt.close(plot.fig)
            return
        # Drain queued callbacks frequently; limit work to keep GUI responsive.
        deadline = time.monotonic() + 0.005
        for _ in range(100):
            rclpy.spin_once(node, timeout_sec=0.0)
            if time.monotonic() >= deadline:
                break

    poll_timer = plot.fig.canvas.new_timer(interval=10)
    poll_timer.add_callback(poll_ros)
    redraw_timer = plot.fig.canvas.new_timer(interval=max(1, int(1000/options.refresh)))
    redraw_timer.add_callback(plot.redraw)
    poll_timer.start()
    redraw_timer.start()
    node.get_logger().info('Viewer ready. Start path_planning with --debug-tracking.')
    try:
        plt.show()
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


if __name__ == '__main__':
    main()
