import argparse
import configparser
import json
import math
import os
import time

import numpy as np
import heapq
import rclpy
from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity

from rclpy.qos import (
    ReliabilityPolicy,
    QoSProfile,
)
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from PIL import Image
from geometry_msgs.msg import Twist


np.set_printoptions(
    2, suppress=True, threshold=np.inf
)  # Print numpy arrays to specified d.p., suppress scientific notation (e.g. 1e-5), and do not truncate

set_logger_level("waypoint", level=LoggingSeverity.DEBUG) # Configure to either LoggingSeverity.INFO or LoggingSeverity.DEBUG

occupancy_grid_resolution = 0.2 # Sim grid resolution, in metres per cell (the real maze uses the same 0.2m cells, see ca2_irl_layout.json)
max_translate_velocity = 1.4 # Overwritten in main() based on sim vs real-life; 0.4m/s cap for real life, please keep that in place

_PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))


# --- Coordinate conversion --------------------------------------------------
# A grid index (i, j) represents a CELL, not a point. That cell's world
# coordinate is its CENTER, e.g. cell [0, 0] is centred half a resolution-step
# away from the grid's origin corner, not exactly on it. This matches how the
# Gazebo world and the real maze are physically laid out (goal tape/markers
# sit in the middle of a cell, not on its boundary line).
def grid_to_world(i:int, j:int, origin:tuple, resolution:float=occupancy_grid_resolution) -> tuple:
    '''Convert grid index (i, j) to the world (x, y) coordinate of that cell's centre.'''
    return (origin[0] + (i + 0.5) * resolution, origin[1] + (j + 0.5) * resolution)

def world_to_grid(x:float, y:float, origin:tuple, resolution:float=occupancy_grid_resolution) -> tuple:
    '''Convert a world (x, y) coordinate to the grid index (i, j) of the cell containing it.'''
    return (int(np.floor((x - origin[0]) / resolution)), int(np.floor((y - origin[1]) / resolution)))


# --- Sim / real-life profiles -----------------------------------------------
# Run without arguments for the simulation. For the real maze pass
# --run test1 | test2 | full  (see README). Everything that differs between
# sim and real -- which map to load, the cell resolution, where the grid's
# [0, 0] corner sits in the frame your pose is reported in, the goal points,
# the speed cap -- lives in one of these profiles.
sim_config = {
    "map_file": "ca2_sim_map.npy",
    "origin": (-1.0, -5.0),
    "resolution": occupancy_grid_resolution,
    "goal_list": [(3.5, -3.5), (3.3, 0.3), (2.5, -3.5), (-0.3, -3.7)],
    "max_translate_velocity": 1.4,
}

IRL_RUNS = ("test1", "test2", "full")


# --- Real maze <-> Optitrack frame -------------------------------------------
# The real maze is a fixed layout (ca2_irl_map.npy + ca2_irl_layout.json, in
# "maze-local" metres: (0, 0) is the outer corner of array cell [0, 0], +x runs
# along array axis 0, +y along axis 1 -- exactly like the sim grid). Optitrack
# reports poses in ITS OWN frame, wherever the maze happens to be taped down.
# optitrack_variables.config holds the 3 numbers that relate the two:
#   origin_x, origin_y : Optitrack coordinates of the maze's (0, 0) corner
#   rotation_deg       : angle of the maze's +x axis in the Optitrack frame
#                        (counter-clockwise positive, degrees)
# The pose from Optitrack is converted into the maze frame before your code
# sees it, so self.origin is (0, 0) and everything else (world_to_grid,
# goal_list, ...) works exactly like in simulation.
def wrap_deg(angle:float) -> float:
    '''Wrap an angle in degrees into [-180, 180).'''
    return (angle + 180.0) % 360.0 - 180.0

def optitrack_to_maze(x:float, y:float, heading_deg:float, frame:dict) -> tuple:
    '''Optitrack-frame pose (x, y, heading in degrees) -> maze-frame pose.'''
    dx, dy = x - frame["origin_x"], y - frame["origin_y"]
    r = math.radians(frame["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (c * dx + s * dy, -s * dx + c * dy,
            wrap_deg(heading_deg - frame["rotation_deg"] - frame.get("heading_offset_deg", 0.0))) # offset: the Motive rigid body's "forward" vs the robot's front

def maze_to_optitrack(mx:float, my:float, heading_deg:float, frame:dict) -> tuple:
    '''Inverse of optitrack_to_maze (handy for working out where to place the robot).'''
    r = math.radians(frame["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (frame["origin_x"] + c * mx - s * my, frame["origin_y"] + s * mx + c * my,
            wrap_deg(heading_deg + frame["rotation_deg"] + frame.get("heading_offset_deg", 0.0)))

def robot_frame(config:dict, robot_number:int) -> dict:
    '''The frame for one robot: the shared origin/rotation plus that robot's heading offset (per-robot value from the config if listed).'''
    frame = dict(config["frame"])
    frame["heading_offset_deg"] = config["heading_offsets"].get(robot_number, frame["heading_offset_deg"])
    return frame

def load_irl_config(run:str) -> dict:
    '''Build the real-maze profile for run "test1", "test2" or "full".'''
    if run not in IRL_RUNS:
        raise ValueError(f"--run must be one of {IRL_RUNS}, got {run!r}")
    with open(os.path.join(_PACKAGE_DIR, "ca2_irl_layout.json")) as f:
        layout = json.load(f)
    parser = configparser.ConfigParser()
    config_path = os.path.join(_PACKAGE_DIR, "optitrack_variables.config")
    if not parser.read(config_path):
        raise FileNotFoundError(f"Could not read {config_path}")
    frame = {
        "origin_x": parser.getfloat("frame", "origin_x"),
        "origin_y": parser.getfloat("frame", "origin_y"),
        "rotation_deg": parser.getfloat("frame", "rotation_deg"),
    }
    # Rigid bodies are created in Motive with the robot facing the maze's +y direction, so the rigid body's zero heading IS maze +y
    # (maze heading 90 deg): heading_maze = reported + 90 = reported - rotation - offset  =>  offset = -(rotation + 90). "auto" means that.
    raw_offset = parser.get("frame", "heading_offset_deg", fallback="auto").strip().lower()
    frame["heading_offset_deg"] = wrap_deg(-(frame["rotation_deg"] + 90.0)) if raw_offset == "auto" else float(raw_offset)
    heading_offsets = {int(k): float(v) for k, v in parser.items("heading_offsets")} if parser.has_section("heading_offsets") else {}
    return {
        "map_file": "ca2_irl_map.npy",
        "origin": (0.0, 0.0), # pose is converted into the maze frame, so the grid corner is (0, 0)
        "resolution": float(layout["resolution"]),
        "goal_list": [tuple(g) for g in layout["runs"][run]["goals"]],
        "start": tuple(layout["runs"][run]["start"]),
        "frame": frame,
        "robot_number": parser.getint("robot", "number"),
        "heading_offsets": heading_offsets, # per-robot override of frame["heading_offset_deg"]
        "max_translate_velocity": 0.4, # Please keep this in place
    }

# OUR GROUP'S ASTAR ALGO
def heuristic(cell, goal):
    """Manhattan distance for four-direction movement."""
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

def astar(grid, start, goal):
    """Return a shortest grid path from start to goal, or None."""
    rows, cols = grid.shape

    def is_free(cell):
        i, j = cell
        # Check bounds first; negative NumPy indices wrap around.
        return (
            0 <= i < rows
            and 0 <= j < cols
            and grid[i, j] == 0
        )

    if not is_free(start) or not is_free(goal):
        return None

    open_set = {start}   # Discovered cells waiting for expansion
    closed_set = set()   # Cells already expanded
    g = {start: 0}       # Best known cost from start to each cell
    parent = {}         # Previous cell on the best known route

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while open_set:
        # Select the cell with the lowest total estimated cost.
        current = min(
            open_set,
            key=lambda cell: g[cell] + heuristic(cell, goal)
        )
        open_set.remove(current)

        if current == goal:
            # Follow parents from goal to start, then reverse.
            path = [current]
            while current != start:
                current = parent[current]
                path.append(current)
            return path[::-1]

        closed_set.add(current)

        for di, dj in directions:
            neighbour = (current[0] + di, current[1] + dj)

            if not is_free(neighbour) or neighbour in closed_set:
                continue

            # One horizontal or vertical step costs 1.
            new_g = g[current] + 1

            # Compare the incoming direction (previous → current) with the
            # outgoing direction (current → neighbour). Skip the starting cell.
            if current in parent and (di, dj) != (current[0] - parent[current][0], current[1] - parent[current][1]):
                # Penalise direction changes to favour paths with fewer turns.
                new_g += 0.1

            # Update if this route is cheaper.
            # Undiscovered cells have a default cost of infinity.
            if new_g < g.get(neighbour, float("inf")):
                g[neighbour] = new_g
                parent[neighbour] = current
                open_set.add(neighbour)

    return None  # No route exists

def wrap_angle(angle):
    """Wrap a RADIAN angle to [-pi, pi), giving the shortest turn."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def get_heading_error(robot_pose, target):
    """
    robot_pose: [x, y, heading_in_DEGREES], as stored in self.pose
    target: [x, y] in WORLD coordinates

    Returns the heading error in RADIANS.
    """

    # Calculate the direction from the robot toward the target.
    dx = target[0] - robot_pose[0]
    dy = target[1] - robot_pose[1]

    # atan2 handles all quadrants and returns an angle in radians.
    desired_heading = np.arctan2(dy, dx)

    # Convert the supplied robot heading from degrees to radians.
    current_heading = np.deg2rad(robot_pose[2])

    # Positive error: turn anticlockwise.
    # Negative error: turn clockwise.
    return wrap_angle(desired_heading - current_heading)

# PID CLASS

# Checker if it reached grid
def has_reached_grid(robot_pose, target_cell, origin,
                     resolution=occupancy_grid_resolution, tolerance=0.05) -> bool:
    """Return True when BOTH x and y are within tolerance of the cell centre.

    robot_pose: world [x, y] or [x, y, heading]; None means no pose yet.
    target_cell: grid indices (i, j). Tolerance is in metres, inclusive.
    This checks a square around the centre, not a circular distance.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if robot_pose is None:
        return False

    centre_x, centre_y = grid_to_world(*target_cell, origin, resolution)
    return bool(
        abs(robot_pose[0] - centre_x) <= tolerance
        and abs(robot_pose[1] - centre_y) <= tolerance
    )

# SIMPLIFY Waypoints
def get_turn_waypoints(path):
    """Keep the start, every direction-change cell, and the end."""

    # Empty, one-cell, and two-cell paths need no simplification.
    if len(path) <= 2:
        return list(path)

    waypoints = [path[0]]

    # Inspect each cell that has both a previous and a next cell.
    for i in range(1, len(path) - 1):
        previous = path[i - 1]
        current = path[i]
        following = path[i + 1]

        # Direction taken to ENTER the current cell.
        incoming = (
            current[0] - previous[0],
            current[1] - previous[1]
        )

        # Direction taken to LEAVE the current cell.
        outgoing = (
            following[0] - current[0],
            following[1] - current[1]
        )

        # Same direction: current lies within a straight run; skip it.
        # Different direction: current is a turn point; keep it.
        if incoming != outgoing:
            waypoints.append(current)

    waypoints.append(path[-1])
    return waypoints


def plot_errors(log, filename="pid_errors.png"):
    """log = list of (time, heading_error_deg, lateral_error_m, turn_cmd, forward_cmd)"""
    if not log:
        print("Nothing to plot")
        return

    import matplotlib.pyplot as plt
    plt.close("all")               # close the live window
    plt.switch_backend("Agg")      # then save to file quietly (and delete the "import matplotlib" line above them if it's there)

 
    t, head, lat, turn, fwd = zip(*log)
 
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
 
    axs[0].plot(t, head, color="tab:red")
    axs[0].axhline(0, color="k", lw=0.5)
    axs[0].set_ylabel("Heading error (deg)")
 
    axs[1].plot(t, lat, color="tab:blue")
    axs[1].axhline(0, color="k", lw=0.5)
    axs[1].set_ylabel("Lateral error (m)")
 
    axs[2].plot(t, turn, label="turn (rad/s)", color="tab:green")
    axs[2].plot(t, fwd, label="forward (m/s)", color="tab:orange")
    axs[2].axhline(0, color="k", lw=0.5)
    axs[2].set_ylabel("Commands")
    axs[2].set_xlabel("Time (s)")
    axs[2].legend()
 
    axs[0].set_title("Robot tracking errors")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}")

class LivePlot:
#     """Pop-up window that updates while the robot moves (shows the last ~20 s)."""
 
    def __init__(self, n_points=400):          # 400 ticks at 20 Hz = 20 seconds
        import matplotlib.pyplot as plt
        plt.ion()                              # interactive mode: window stays open and updates
        self.n_points = n_points
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        self.line1, = self.ax1.plot([], [], color="tab:red")
        self.line2, = self.ax2.plot([], [], color="tab:blue")
        self.ax1.set_ylabel("Heading error (deg)")
        self.ax2.set_ylabel("Lateral error (m)")
        self.ax2.set_xlabel("Time (s)")
        self.ax1.set_title("Live tracking errors")
        for ax in (self.ax1, self.ax2):
            ax.axhline(0, color="k", lw=0.5)
        self.fig.tight_layout()

    def update(self, log):
        """log rows are (time, heading_error_deg, lateral_error_m, turn, forward)."""
        if len(log) < 2:
            return
        recent = log[-self.n_points:]
        t = [r[0] for r in recent]
        self.line1.set_data(t, [r[1] for r in recent])
        self.line2.set_data(t, [r[2] for r in recent])
        for ax in (self.ax1, self.ax2):
            ax.relim()
            ax.autoscale_view()
        self.ax1.set_xlim(t[0], max(t[-1], t[0] + 1))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


class WaypointNode(Node):
    '''Node to calculate path and move robot towards given goal_coordinates, using pose info from either gazebo odometer or optitrack'''
    def __init__(self, map_array:np.array, goal_list:list, is_simulation:bool=True, origin:tuple=(0.0, 0.0), resolution:float=occupancy_grid_resolution,
                 frame:dict=None, robot_number:int=3, expected_start:tuple=None, calibrate:bool=False):
        super().__init__('waypoint')
        self.get_logger().info("Starting WaypointNode")

        # Our own arguemnts
        self.is_simulation = is_simulation
        self.plan_new_path = True  # Flag to start path planning
        self.goal_idx = 0
        self.waypoints = []

        self.log = []                  # stores errors for the report graph
        self.t0 = time.time()

        try:
            self.live = LivePlot()
        except Exception as e:                  # e.g. no display / no Tk installed
            self.live = None
            self.get_logger().warn(f"Live plot disabled: {e}")
        self.create_timer(0.2, self.update_live_plot)   # redraw 5 times per second

        
        # Controller Settings
        self.lookahead_cells = 2
        self.waypoint_path_indices = []
        self.progress_idx = 0

        # Start with logging only. Set False after checking target selection.
        self.tracking_test_only = False

        # Remember whether lateral correction is active.
        self.correcting_lateral = False  

        # Controller settings — conservative starting values.
        self.mode = "ALIGN"
        self.forward_speed = 0.50           # Maximum forward speed, m/s.
        self.turn_speed = 2.8 if self.is_simulation else 0.4
        self.min_turn_speed = 1.5 if self.is_simulation else 0.0
        self.heading_tolerance = np.deg2rad(0.3)
        self.heading_kp = 3.5
        self.distance_kp = 1.5
        self.realign_angle = np.deg2rad(30.0)


        # Subscribe to the dynamic_pose topic from Gazebo that publishes ground-truth pose data
        if self.is_simulation:
            self.subscription = self.create_subscription(Odometry, 'odom', self.odometer_callback, 2)
        else:
            qos_profile = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)

            self.map_sub = self.create_subscription(
                PoseStamped,
                f'/vrpn_mocap/bingda_{robot_number:03d}/pose',
                self.optitrack_callback,
                qos_profile
                )

        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10) # Publish to cmd_vel node
        self.timer = self.create_timer(0.05, self.timer_callback)  # Runs at 20Hz. Can be changed.

        self.goal_list = goal_list
        self.map_array = map_array
        self.origin = origin # World (x, y) coordinate of the grid's [0, 0] corner. Use with grid_to_world()/world_to_grid()
        self.resolution = resolution # Metres per grid cell (0.2). Use with grid_to_world()/world_to_grid()

        self.frame = frame # Optitrack->maze frame (real robot only), see optitrack_to_maze()
        self.expected_start = expected_start # Where this run says the robot should be placed (maze frame), real robot only
        self.calibrate = calibrate # Real robot: only print poses, never publish cmd_vel
        self.raw_pose = None # Latest pose exactly as Optitrack reported it (real robot only)
        self._start_checked = False
        self._calibrate_ticks = 0

        self.pose = None
        self.path = [] # Set this to your planned route (a list of grid-index tuples, in travel order) once you've computed it -- it'll automatically show up in the terminal map print
        self._last_printed_path = None

    def _print_calibration(self):
        '''--calibrate: twice a second, print the raw Optitrack pose next to the maze-frame pose and grid cell.'''
        self._calibrate_ticks += 1
        if self._calibrate_ticks % 10:
            return
        ox, oy, oh = self.raw_pose
        mx, my, mh = self.pose
        i, j = world_to_grid(mx, my, self.origin, self.resolution)
        self.get_logger().info(f"optitrack ({ox:+.3f}, {oy:+.3f}, {oh:+7.1f} deg)  ->  maze ({mx:+.3f}, {my:+.3f}, {mh:+7.1f} deg)  cell [{i}, {j}]")

    def _check_start_once(self):
        '''Once, on the first pose: warn if the robot isn't near this run's start point (usually a placement or config mistake).'''
        if self._start_checked or self.expected_start is None:
            return
        self._start_checked = True
        dist = math.hypot(self.pose[0] - self.expected_start[0], self.pose[1] - self.expected_start[1])
        msg = f"Robot is at maze ({self.pose[0]:.2f}, {self.pose[1]:.2f}); this run starts at {self.expected_start} -- {dist:.2f} m away"
        if dist > 0.4:
            self.get_logger().warn(msg + ". Check where the robot is placed; if it is on the start point, ask a TA.")
        else:
            self.get_logger().info(msg + " (ok)")

    def print_map(self):
        '''Prints the occupancy grid to the terminal: walls, your current position ('S'), all goal points ('W'/'G'),
        and your planned route (self.path) if you've set one ('*'). Safe to call anytime pose is known; does nothing
        useful before then. Called automatically from timer_callback() whenever self.path changes.'''
        if self.pose is None:
            return
        shape = self.map_array.shape
        clip = lambda cell: (int(np.clip(cell[0], 0, shape[0]-1)), int(np.clip(cell[1], 0, shape[1]-1)))
        current_cell = clip(world_to_grid(self.pose[0], self.pose[1], self.origin, self.resolution))
        goal_cells = [clip(world_to_grid(gx, gy, self.origin, self.resolution)) for gx, gy in self.goal_list]
        grid = Grid(self.map_array, starting_position=current_cell, goal_position=goal_cells[-1])
        #grid.print_grid_map(waypoints=goal_cells, path=self.path)
        grid.draw_grid_map(waypoints=self.waypoints, path=self.path)

    def yaw_from_quaternion(self, q):
        '''Returns yaw angle (in rad) for orientation based on given quaternion input q'''
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def optitrack_callback(self, msg:PoseStamped):
        '''Callback to calculate 2D pose info from Optitrack node. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        x, y = msg.pose.position.x, msg.pose.position.y
        heading = np.rad2deg(self.yaw_from_quaternion(msg.pose.orientation))
        self.raw_pose = np.array((x,y,heading))
        if self.frame is not None:
            x, y, heading = optitrack_to_maze(x, y, heading, self.frame) # Your code works in the maze frame, like in simulation
        self.pose = np.array((x,y,heading))
        return self.pose

    def odometer_callback(self, msg):
        '''Callback to calculate 2D pose info from Gazebo odomoter. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        latest_pose_msg = msg.pose.pose
        heading = np.rad2deg(self.yaw_from_quaternion(latest_pose_msg.orientation))
        self.pose = np.array((latest_pose_msg.position.x, latest_pose_msg.position.y, heading))
        return self.pose

    def move_2D(self, x:float=0.0, y:float=0.0, turn:float=0.0):
        '''Publishes a Twist message to ROS to move a robot. Inputs are x and y linear velocities, as well as turn (z-axis yaw) angular velocity.'''
        twist_msg = Twist()
        x = np.clip(x, -max_translate_velocity, max_translate_velocity)
        y = np.clip(y, -max_translate_velocity, max_translate_velocity)
        turn = np.clip(turn, -max_translate_velocity*2, max_translate_velocity*2)
        twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z = float(x), float(y), 0.0
        twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z = 0.0, 0.0, float(turn)
        self.publisher_.publish(twist_msg)

    def set_waypoints(self, waypoints:list):
        '''Set new waypoints when a goal has been reached'''
        self.goal_reached = False
        self.waypoints = waypoints

    ############# Our own functions ################
    def plan_current_goal(self):
        """Plan the route to the current goal and prepare its waypoints."""
        self.plan_new_path = False # Do not plan new plan until goal has been reached
        self.move_2D()

        # Convert the current robot position into a grid cell.
        start = world_to_grid(self.pose[0], self.pose[1],self.origin, self.resolution)

        # Choose current goal, convert into grid coordinates
        goal_x, goal_y = self.goal_list[self.goal_idx]
        goal = world_to_grid(goal_x, goal_y,self.origin, self.resolution)

        # PLAN A* ROUTE
        route = astar(self.map_array, start, goal)
        if route is None:
            self.get_logger().error(
                f"No route found: {start} -> {goal}"
            )
        else:
            # Route found!
            self.path = route
            self.waypoints = get_turn_waypoints(route) # Navigate using only the start, corners, and endpoint.
            self.waypoint_path_indices = [self.path.index(cell) for cell in self.waypoints] # Link each corner/endpoint to its position in the full path.
            self.progress_idx = 0
            self.mode = "ALIGN"
            self.get_logger().info(
                f"Route to Goal {goal_x,goal_y} found! {len(route) - 1} steps, "
                f"{len(self.waypoints)} simplified waypoints"
            )
            #  Initialise waypointskip the starting cell unless it is the only cell in the route.
            self.current_waypoint_idx = 1 if len(self.waypoints) > 1 else 0 

    def get_lookahead_info(self):
        """Use the full path for lookahead and simplified waypoints for corners."""
        waypoint_idx = self.current_waypoint_idx

        lateral_error = 0.0  # Signed distance from the current segment; metres.

        end_idx = self.waypoint_path_indices[waypoint_idx]
        start_idx = self.waypoint_path_indices[max(0, waypoint_idx - 1)]

        start = np.array(grid_to_world(
            *self.path[start_idx], self.origin, self.resolution
        ))
        end = np.array(grid_to_world(
            *self.path[end_idx], self.origin, self.resolution
        ))

        robot = self.pose[:2]
        segment = end - start
        length = np.linalg.norm(segment)

        if length > 1e-9:
            direction = segment / length

            offset = robot - start
            lateral_error = direction[0] * offset[1] - direction[1] * offset[0]

            # Progress depends on movement along the line, not sideways movement.
            distance_along = float(np.clip(
                np.dot(robot - start, direction),
                0.0,
                length,
            ))

            closest_point = start + distance_along * direction

            # Count cell centres passed along this straight segment.
            steps = int(np.floor(
                distance_along / self.resolution + 1e-9
            ))

            # Keep progress in order and within the current segment.
            self.progress_idx = min(
                end_idx,
                max(start_idx, self.progress_idx, start_idx + steps),
            )
        else:
            # One-cell route: approach that cell's centre directly.
            closest_point = robot.copy()
            self.progress_idx = end_idx

        # Look two cells ahead, but never beyond the next corner.
        lookahead_idx = min(
            self.progress_idx + self.lookahead_cells,
            end_idx,
        )

        target = np.array(grid_to_world(
            *self.path[lookahead_idx], self.origin, self.resolution
        ))

        # Heading correction points from the robot toward the target.
        delta = target - robot
        heading_error = 0.0
        use_lookahead = False

        if np.linalg.norm(delta) > 1e-9:
            if abs(lateral_error) >= 0.03:
                self.correcting_lateral = True
            elif abs(lateral_error) <= 0.02:
                self.correcting_lateral = False

            # Aim at the lookahead when correcting or approaching the endpoint.
            use_lookahead = self.correcting_lateral or lookahead_idx == end_idx
            # If not correcting or reaching endpoint, follow direction of segment
            desired_heading = np.arctan2(delta[1], delta[0]) if use_lookahead else np.arctan2(segment[1], segment[0])

            # Shortest turn from current heading: positive anticlockwise, negative clockwise.
            heading_error = wrap_angle(
                desired_heading - np.deg2rad(self.pose[2])
            )

        return {
            "heading_error": float(heading_error),
            "distance_to_lookahead": float(
                np.linalg.norm(target - robot)
            ),
            "cell_number": self.progress_idx - start_idx + 1,
            "cell_count": end_idx - start_idx + 1,
            "lookahead": use_lookahead,
            "lateral_error": float(lateral_error)
        }

    def calculate_motion(self, heading_error, distance_to_lookahead):
        """Forward P control uses the distance from the robot to the lookahead target."""

        # ALIGN finishes at a small error.
        # DRIVE only returns to ALIGN if the error becomes large.
        if self.mode == "ALIGN":
            if abs(heading_error) <= self.heading_tolerance:
                self.mode = "DRIVE"
        elif abs(heading_error) >= self.realign_angle:
            self.mode = "ALIGN"

        # Heading P controller.
        turn = self.heading_kp * heading_error

        if abs(heading_error) <= self.heading_tolerance:
            turn = 0.0
        else:
            # Simulation workaround: preserve direction when boosting small turns.
            turn = np.copysign(max(abs(turn), self.min_turn_speed), turn)  # Boost small turns while preserving direction.
            turn = float(np.clip(turn, -self.turn_speed, self.turn_speed))  # Keep the maximum speed limit.
        forward = 0.0

        if self.mode == "DRIVE":
            # Forward P controller: closest point to lookahead target.
            forward = min(
                self.forward_speed,
                self.distance_kp * distance_to_lookahead,
            )

            # Reduce forward movement while correcting heading.
            forward *= max(0.0, np.cos(heading_error))

            # Temporary simulation test: determine whether low forward commands cause the stall.
            if self.is_simulation:
                forward = min(self.forward_speed, max(forward, 0.15))

        return self.mode, float(forward), float(turn)

    def advance_waypoint(self):
        """Called only after the current waypoint has been reached"""
        self.mode = "ALIGN"  # Align with the new segment before driving.
        self.current_waypoint_idx += 1
        self.move_2D()  # Stop briefly before checking the next waypoint.

        # Last waypoint reached
        if self.current_waypoint_idx >= len(self.waypoints):
            if self.goal_idx == len(self.goal_list) - 1:
            # Last goal reached: keep the goal index unchanged.
                self.plan_new_path = False
                self.get_logger().info("All goals reached!")
            else:
                # Another goal remains. Plan its route next callback.
                self.goal_idx += 1
                self.plan_new_path = True

    def update_live_plot(self):
        if self.live is not None:
            self.live.update(self.log)


    def timer_callback(self):
        """Controller loop. Insert path planning and PID control logic here"""
        if self.pose is None:
            return # Does not run if no pose received from Odom or Optitrack
        now = time.time()
        if now - getattr(self, "_last_pose_log", 0.0) >= 1.0: # at most once a second, so it does not bury the map / warnings below
            self._last_pose_log = now
            self.get_logger().debug(f"Pose: {self.pose}")

        if self.calibrate:
            self._print_calibration()
            return # Calibration mode: look, don't drive

        self._check_start_once()

        if self.path != self._last_printed_path: # Prints once immediately (map + start + goals), then again each time self.path changes
            self.print_map()
            self._last_printed_path = list(self.path)

        ###### INSERT CODE HERE ######
        # Plan with A* and Generate Waypoints for each goal
        if self.plan_new_path:
            self.plan_current_goal()

        # Complete WayPoints
        if self.pose is None or not self.waypoints: # Wait until we have a robot pose and a planned path.
            self.move_2D()
            return

        # Stop when all waypoints have been reached.
        if self.current_waypoint_idx >= len(self.waypoints):
            self.move_2D()
            return

        # Get the grid cell we are currently trying to reach.
        endpoint_cell = self.waypoints[self.current_waypoint_idx]

        # Check whether BOTH x and y are within 5 cm of its centre.
        if has_reached_grid(self.pose,endpoint_cell,self.origin,self.resolution,tolerance=0.05,):
            self.advance_waypoint()
            return # end function call to either plan new path or stop if all goals have been reached


        tracking = self.get_lookahead_info()

        # Endpoint recovery is deferred: stop and report this test failure.
        if tracking["distance_to_lookahead"] <= 1e-6:
            self.move_2D()
            self.get_logger().warn(
                "STOP: no forward distance remains, but endpoint is not reached.",
                throttle_duration_sec=1.0,
            )
            return

        mode, forward, turn = self.calculate_motion(
            tracking["heading_error"],
            tracking["distance_to_lookahead"],
        )

        self.log.append((time.time() - self.t0,
                          np.rad2deg(tracking["heading_error"]),
                          tracking["lateral_error"],
                          turn, forward))


        # Stage 1 prints tracking information without moving the robot.
        if self.tracking_test_only:
            mode = "TEST"
            forward = 0.0
            turn = 0.0

        lookahead_status = "YES" if tracking["lookahead"] else "NO"

        self.get_logger().info(
            f"{mode} | GOAL {self.goal_idx + 1}/{len(self.goal_list)} | "
            f"CELL {tracking['cell_number']}/{tracking['cell_count']} | "
            f"LOOKAHEAD {lookahead_status} | "
            f"DISTANCE TO LOOKAHEAD {tracking["distance_to_lookahead"]:+.3f} | "
            f"LATERAL {tracking['lateral_error']:+.3f} m | "
            f"HEADING {self.pose[2]:.1f} deg | "
            f"CORRECT {np.rad2deg(tracking['heading_error']):+.1f} deg | "
            f"FORWARD {forward:.2f} m/s | TURN {turn:.2f} rad/s\n",
            throttle_duration_sec=1.0,
        )

        self.move_2D(x=forward, y=0.0, turn=turn)
        

        ###### INSERT CODE HERE ######


class Grid():
    '''
    Grid class to use with occupancy grid. Contains the following functions:
        check_grid_validity : Uses flood fill to check if there's a valid path from start to goal position
        draw_grid_map : Creates a colour image of the grid, as well as waypoints and full solution path if given. Can show it and/or save it to a .png file
        print_grid_map : Prints an ASCII-art version of the same map (grid, waypoints, path) to the terminal
        animate_path : Prints the ASCII-art map frame by frame in the terminal, animating the robot moving along a given path

    __init__ input Args:
        grid_array : 2D numpy array representing the occupancy grid
        starting_position : tuple of starting indices within the numpy array
        goal_position : tuple of goal indices within the numpy array. Works with negative indices as well
    '''
    def __init__(self, grid_array:np.array=np.array([]), starting_position:tuple=(0,0), goal_position:tuple=(-1,-1)):
        self.grid = grid_array
        self.shape = self.grid.shape
        self.starting_position = starting_position

        # If goal position given with negative index, need to convert to +ve
        if goal_position[0] < 0:
            goal_x = self.shape[0] + goal_position[0]
        else:
            goal_x = goal_position[0]
        if goal_position[1] < 0:
            goal_y = self.shape[1] + goal_position[1]
        else:
            goal_y = goal_position[1]

        self.goal_position = (goal_x, goal_y)

    def check_grid_validity(self):
        '''Use flood fill to check if there's a viable path between start and goal positions'''
        grid = self.grid.copy()
        flood_stack = [(self.goal_position[0], self.goal_position[1])]
        while flood_stack:
            tile = flood_stack[0]
            del flood_stack[0]
            try:
                next_tile = (tile[0]+1, tile[1])
                if grid[next_tile] == 0:
                    grid[next_tile] = 1
                    flood_stack.append(next_tile)
            except:pass
            try:
                next_tile = (tile[0]-1, tile[1])
                if grid[next_tile] == 0:
                    grid[next_tile] = 1
                    flood_stack.append(next_tile)
            except:pass
            try:
                next_tile = (tile[0], tile[1]+1)
                if grid[next_tile] == 0:
                    grid[next_tile] = 1
                    flood_stack.append(next_tile)
            except:pass
            try:
                next_tile = (tile[0], tile[1]-1)
                if grid[next_tile] == 0:
                    grid[next_tile] = 1
                    flood_stack.append(next_tile)
            except:pass
        if grid[self.starting_position] == 1: # Means the flood is able to reach starting position from the ending position
            return True
        else:
            return False

    def _colour_grid(self, waypoints:list=(), path:list=(), obstacle_threshold:float=50) -> np.array:
        '''Builds the (H, W, 3) colour image array shared by draw_grid_map. Maze walls in blue, empty space in white, path taken in green and waypoints in red'''
        image_grid = np.ones((self.grid.shape[0],self.grid.shape[1],3), dtype=np.uint8)
        image_grid[self.grid <= obstacle_threshold] = (255,255,255)
        image_grid[self.grid > obstacle_threshold] = (0,0,255)

        for x, y in path:
            image_grid[x][y] = (0,255,0)

        for point in waypoints:
            image_grid[point] = (255,0,0)

        return image_grid

    def draw_grid_map(self, waypoints:list=(), path:list=(), obstacle_threshold:float=50, save_path:str=None, show:bool=True):
        '''Creates an image of the maze and path taken. Maze walls in blue, empty space in white, path taken in green and waypoints in red

        Args:
            waypoints : list (or other iterable) of tuple coordinates representing all the grid indices for the waypoints. Will be represented in red, takes precedence over path
            path : list (or other iterable) of tuple coordinates representing all the grid indices forming the solution path. Will be represented in green
            obstacle_threshold : Optional float to indicate threshhold for whether a grid is considered occupied. Not important for ca2
            save_path : Optional file path (e.g. "path_result.png") to save the image to, in addition to/instead of showing it
            show : Whether to pop up the image in a viewer (default True). Set to False if you only want to save it
        '''
        image_grid = self._colour_grid(waypoints, path, obstacle_threshold)

        image_grid = np.flip(image_grid, axis=1)[::-1]
        img = Image.fromarray(image_grid, 'RGB')

        # Resize image
        base_width = 500
        wpercent = (base_width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((base_width, hsize), getattr(Image, "Resampling", Image).NEAREST)

        if save_path:
            img.save(save_path)
            print(f"Saved map image to {save_path}")

        if show:
            img.show()

    def print_grid_map(self, waypoints:list=(), path:list=(), obstacle_threshold:float=50):
        '''Prints an ASCII-art version of the maze to the terminal: '#' wall, '.' free, '*' solution path, 'W' waypoint, 'S' start, 'G' goal

        Args:
            waypoints : list (or other iterable) of tuple coordinates representing all the grid indices for the waypoints
            path : list (or other iterable) of tuple coordinates representing all the grid indices forming the solution path
            obstacle_threshold : Optional float to indicate threshhold for whether a grid is considered occupied. Not important for ca2
        '''
        chars = np.where(self.grid > obstacle_threshold, '#', '.').astype('<U1')

        for x, y in path:
            chars[x, y] = '*'
        for point in waypoints:
            chars[point] = 'W'
        chars[self.starting_position] = 'S'
        chars[self.goal_position] = 'G'

        display = np.flip(chars, axis=1)[::-1]
        print('\n'.join(''.join(row) for row in display))

    def animate_path(self, path:list, waypoints:list=(), obstacle_threshold:float=50, delay:float=0.2):
        '''Animates the robot moving along path, one cell at a time, by reprinting the ASCII-art map to the terminal.
        'R' marks the robot's current cell, '*' cells it has already passed through.

        Args:
            path : list of tuple grid indices forming the solution path, in travel order
            waypoints : list (or other iterable) of tuple coordinates representing all the grid indices for the waypoints
            obstacle_threshold : Optional float to indicate threshhold for whether a grid is considered occupied. Not important for ca2
            delay : Seconds to pause between animation frames
        '''
        path = list(path)
        base_chars = np.where(self.grid > obstacle_threshold, '#', '.').astype('<U1')
        for point in waypoints:
            base_chars[point] = 'W'
        base_chars[self.goal_position] = 'G'

        for step in range(len(path)):
            frame = base_chars.copy()
            for x, y in path[:step]:
                frame[x, y] = '*'
            frame[path[step]] = 'R'

            display = np.flip(frame, axis=1)[::-1]
            print("\033c", end="")  # Clear terminal between frames
            print('\n'.join(''.join(row) for row in display))
            time.sleep(delay)


def main(args=None):
    global max_translate_velocity

    arg_parser = argparse.ArgumentParser(description="RB2301 CA2 path planning")
    arg_parser.add_argument(
        '--run', choices=IRL_RUNS, default=None,
        help="Run on the real maze: test1, test2 or full (starts/goals in README). Omit to run in Gazebo simulation."
    )
    arg_parser.add_argument(
        '--robot', type=int, default=None,
        help="Number of your Bingda robot (3 -> /vrpn_mocap/bingda_003/pose). Default: your ROS_DOMAIN_ID (set to the robot number on the robots), else [robot] number in optitrack_variables.config."
    )
    arg_parser.add_argument(
        '--calibrate', action='store_true',
        help="Real robot: only print Optitrack and maze-frame poses (twice a second) and never publish cmd_vel. Use it to set origin/rotation."
    )
    cli_args, ros_args = arg_parser.parse_known_args(args=args)
    run = cli_args.run or ("test1" if cli_args.calibrate else None)
    is_simulation = run is None # Remember: pass --run test1|test2|full to ca2.sh when testing on the real lab setup

    print("Starting path planning")
    rclpy.init(args=ros_args)

    robot_number = 3
    frame = expected_start = None
    if is_simulation:
        config = sim_config
    else:
        config = load_irl_config(run)
        frame, expected_start = config["frame"], config["start"]
        domain = os.environ.get("ROS_DOMAIN_ID", "")
        if cli_args.robot is not None:
            robot_number = cli_args.robot
        elif domain.isdigit() and 1 <= int(domain) <= 99: # on the Bingda robots ROS_DOMAIN_ID is the robot number
            robot_number = int(domain)
        else:
            robot_number = config["robot_number"]
        frame = robot_frame(config, robot_number)
        expected_start = config["start"]
        print(f"Running on the real maze, run={run}, robot=bingda_{robot_number:03d}, resolution={config['resolution']}")
        print(f"  maze origin in Optitrack frame = ({frame['origin_x']}, {frame['origin_y']}), rotation = {frame['rotation_deg']} deg, robot heading offset = {frame['heading_offset_deg']} deg")
        print(f"  start={expected_start}, goals={config['goal_list']}")
        if cli_args.calibrate:
            print("  CALIBRATE MODE: printing poses only, the robot will not be commanded")

    max_translate_velocity = config["max_translate_velocity"]

    map_array = np.load(os.path.join(_PACKAGE_DIR, config["map_file"]), allow_pickle=True)
    waypoint = WaypointNode(map_array, config["goal_list"], is_simulation, config["origin"], config["resolution"],
                            frame=frame, robot_number=robot_number, expected_start=expected_start, calibrate=cli_args.calibrate)

    # Start spinning the waypoint node and only stop once SystemExit error is raised within the node callback
    try:
        rclpy.spin(waypoint)
    except (SystemExit, KeyboardInterrupt):    # <-- KeyboardInterrupt = you pressing Ctrl+C
        print("Shutting down")
    finally:
        plot_errors(waypoint.log)              # <-- saves pid_errors.png

    waypoint.destroy_node()
    if rclpy.ok():                             # <-- avoids a crash if Ctrl+C already shut ROS down
        rclpy.shutdown()

if __name__ == '__main__':
    main()