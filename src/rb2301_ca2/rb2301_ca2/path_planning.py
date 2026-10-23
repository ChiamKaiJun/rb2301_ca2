import argparse
import configparser
import json
import math
import os
import time

import numpy as np
from rb2301_ca2.pid_controller import PIDController
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
    """Euclidean straight-line distance to the goal."""
    return math.hypot(
    goal[0] - cell[0],
    goal[1] - cell[1]
    )


def astar(grid, start, goal):
    """Eight-direction A* with a penalty for sharp turns."""
    rows, cols = grid.shape
    turn_weight = 0.1  # Penalty for one 90-degree turn

    def is_free(cell):
        i, j = cell
        return (
            0 <= i < rows
            and 0 <= j < cols
            and grid[i, j] == 0
        )

    if not is_free(start) or not is_free(goal):
        return None

    # Horizontal, vertical and diagonal moves.
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]

    # Each state stores: (i, j, incoming_di, incoming_dj).
    # (0, 0) means no incoming direction at the start.
    start_state = (*start, 0, 0)
    open_set = {start_state}
    closed_set = set()
    g = {start_state: 0.0}
    parent = {}

    while open_set:
        current = min(
            open_set,
            key=lambda s: g[s] + heuristic(s[:2], goal)
        )
        open_set.remove(current)
        i, j, old_di, old_dj = current

        if (i, j) == goal:
            # Return only grid coordinates for your waypoint controller.
            path = [current[:2]]
            while current in parent:
                current = parent[current]
                path.append(current[:2])
            return path[::-1]

        closed_set.add(current)

        for di, dj in directions:
            neighbour = (i + di, j + dj)

            if not is_free(neighbour):
                continue

            # A diagonal needs BOTH neighbouring side cells to be free.
            if di != 0 and dj != 0:
                if (
                    not is_free((i + di, j))
                    or not is_free((i, j + dj))
                ):
                    continue

            next_state = (*neighbour, di, dj)

            if next_state in closed_set:
                continue

            # 1 for horizontal/vertical; sqrt(2) for diagonal.
            step_cost = math.hypot(di, dj)

            turn_penalty = 0.0
            if (old_di, old_dj) != (0, 0):
                turn_angle = abs(wrap_angle(
                    math.atan2(dj, di)
                    - math.atan2(old_dj, old_di)
                ))

                turn_penalty = (
                    turn_weight
                    * (turn_angle / math.radians(90)) ** 2
                )

            new_g = g[current] + step_cost + turn_penalty

            if new_g < g.get(next_state, float("inf")):
                g[next_state] = new_g
                parent[next_state] = current
                open_set.add(next_state)

    return None

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

# All controller tuning lives here. Angles are radians, distances metres.
# These are starting values; verify on the robot before increasing speed.
CONTROL_SETTINGS = {
    "heading_kp": 3.5, "heading_ki": 0.0, "heading_kd": 0.0,
    "align_tolerance": math.radians(3.0),
    "heading_deadband": math.radians(1.0),
    "realign_angle": math.radians(80.0),
    "waypoint_tolerance": 0.015,
    "carrot_spacing": 0.2,
    "distance_kp": 2.0,
    "lateral_on": 0.02, "lateral_off": 0.01,

    "corner_tolerance": 0.1,            # Switch 15 cm before a corner
    "curve_speed": 0.2,                # Start testing at 0.5 m/s 
} #     
SIM_CONTROL = {"forward_speed": 4.0, "turn_speed": 2.8,
               "min_turn_speed": 1.5, "min_forward_speed": 0.15}
REAL_CONTROL = {"forward_speed": 0.4, "turn_speed": 0.4,
                "min_turn_speed": 0.0, "min_forward_speed": 0.0}


# Checker if it reached grid
def has_reached_grid(robot_pose, target_cell, origin,
                     resolution=occupancy_grid_resolution, tolerance=0.02) -> bool:
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

class WaypointNode(Node):
    '''Node to calculate path and move robot towards given goal_coordinates, using pose info from either gazebo odometer or optitrack'''
    def __init__(self, map_array:np.array, goal_list:list, is_simulation:bool=True, origin:tuple=(0.0, 0.0), resolution:float=occupancy_grid_resolution,
                 frame:dict=None, robot_number:int=3, expected_start:tuple=None, calibrate:bool=False, debug_tracking:bool=False):
        super().__init__('waypoint')
        self.get_logger().info("Starting WaypointNode, changed")

        # Our own arguemnts
        self.is_simulation = is_simulation
        self.plan_new_path = True  # Flag to start path planning
        self.goal_idx = 0
        self.waypoints = []
        
        self.settings = {**CONTROL_SETTINGS,
                         **(SIM_CONTROL if is_simulation else REAL_CONTROL)}
        self.heading_pid = PIDController(
            self.settings["heading_kp"], self.settings["heading_ki"],
            self.settings["heading_kd"], self.settings["turn_speed"], angular=True)
        self.mode = "ALIGN"
        self.current_waypoint_idx = 0
        self.carrot_from = 0.0
        self.carrot_to = self.settings["carrot_spacing"]
        self.correcting_lateral = False
        self.tracking_test_only = False
        self._last_control_time = None

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
        self._tracking_debug = None
        self._debug_tracking = None
        self._debug_command = (0.0, 0.0)
        if debug_tracking:
            try:
                from rb2301_ca2.tracking_debug import TrackingDebug
                self._tracking_debug = TrackingDebug(self)
            except Exception as exc:
                self.get_logger().error(f"Tracking debug disabled: {exc}")

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
        if self.is_simulation:
            grid.draw_grid_map(waypoints=self.waypoints, path=self.path)
            grid.print_grid_map(waypoints=goal_cells, path=self.path)
            
        else:
            # Works over SSH without a desktop image viewer.
            grid.draw_grid_map(waypoints=self.waypoints, path=self.path, save_path="/tmp/robot_map.png", show=True)
            grid.print_grid_map(waypoints=goal_cells, path=self.path)

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
        turn = np.clip(turn, -self.settings["turn_speed"], self.settings["turn_speed"])
        twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z = float(x), float(y), 0.0
        twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z = 0.0, 0.0, float(turn)
        self.publisher_.publish(twist_msg)
        self._debug_command = (float(x), float(turn))

    def reset_tracking(self):
        """Start a new segment with fresh controller and carrot state."""
        self.mode = "ALIGN"
        self.heading_pid.reset()
        self.correcting_lateral = False
        self.carrot_from = 0.0
        self.carrot_to = self.settings["carrot_spacing"]
        self._last_control_time = None

    def plan_current_goal(self):
        """Plan once per goal; a failed plan must never reuse an old path."""
        self.move_2D()
        self.plan_new_path = False
        self.path, self.waypoints = [], []
        self.current_waypoint_idx = 0
        start = world_to_grid(*self.pose[:2], self.origin, self.resolution)
        goal = world_to_grid(*self.goal_list[self.goal_idx], self.origin, self.resolution)
        route = astar(self.map_array, start, goal)
        if route is None:
            self.get_logger().error(f"No route: start={start}, goal={goal}. Stopped.")
            return
        self.path = route
        self.waypoints = get_turn_waypoints(route)
        self.current_waypoint_idx = 1 if len(self.waypoints) > 1 else 0
        self.reset_tracking()
        self.get_logger().info(
            f"Goal {self.goal_idx + 1}: {len(route)-1} grid steps, "
            f"{len(self.waypoints)} waypoints")

    def waypoint_reached(self):
        """Corners and goals still require BOTH coordinates within tolerance."""
        idx = self.current_waypoint_idx

        # Intermediate corner: switch before reaching its exact position.
        if idx < len(self.waypoints) - 1:
            target = np.array(grid_to_world(
                *self.waypoints[idx], self.origin, self.resolution
            ))

            distance = np.linalg.norm(target - self.pose[:2])
            return distance <= self.settings["corner_tolerance"]

        return has_reached_grid(
            self.pose, self.waypoints[self.current_waypoint_idx], self.origin,
            self.resolution, tolerance=self.settings["waypoint_tolerance"])

    def get_lookahead_info(self):
        """Advance the carrot at 90% along each carrot interval, never past a corner.

        Progress is projected onto the segment: sideways motion does not count.
        The final carrot stays at the endpoint until waypoint_reached() succeeds.
        """
        idx = self.current_waypoint_idx
        start = np.array(grid_to_world(*self.waypoints[max(0, idx-1)],
                                       self.origin, self.resolution))
        end = np.array(grid_to_world(*self.waypoints[idx], self.origin, self.resolution))
        robot = self.pose[:2]
        segment = end - start
        length = float(np.linalg.norm(segment))
        lateral = 0.0
        target = end
        if length > 1e-9:
            direction = segment / length
            offset = robot - start
            progress = float(np.clip(np.dot(offset, direction), 0.0, length))
            lateral = float(direction[0]*offset[1] - direction[1]*offset[0])
            self.carrot_to = min(
                length, progress + self.settings["carrot_spacing"]
            )
            target = start + self.carrot_to * direction

        # This hysteresis selects carrot aiming versus parallel-to-segment aiming.
        # It is independent of the ALIGN/DRIVE hysteresis in calculate_motion().
        if abs(lateral) >= self.settings["lateral_on"]:
            self.correcting_lateral = True
        elif abs(lateral) <= self.settings["lateral_off"]:
            self.correcting_lateral = False
        at_endpoint = self.carrot_to >= length
        use_carrot = self.correcting_lateral or at_endpoint
        heading_source = "ENDPOINT" if at_endpoint else (
            "CARROT" if self.correcting_lateral else "SEGMENT")
        delta = target - robot
        error = 0.0
        if np.linalg.norm(delta) > 1e-9:
            desired = math.atan2(delta[1], delta[0]) if use_carrot else math.atan2(segment[1], segment[0])
            error = wrap_angle(desired - math.radians(self.pose[2]))
        return {"heading_error": error, "lateral_error": lateral,
                "distance_to_endpoint": float(np.linalg.norm(end-robot)),
                "target": target, "use_carrot": use_carrot,
                "heading_source": heading_source}

    def calculate_motion(self, tracking, dt):
        """State decisions and speed policy; PID only calculates angular output."""
        cfg = self.settings
        error = tracking["heading_error"]
        old_mode = self.mode
        if self.mode == "ALIGN" and abs(error) <= cfg["align_tolerance"]:
            self.mode = "DRIVE"
        elif self.mode == "DRIVE" and abs(error) >= cfg["realign_angle"]:
            self.mode = "ALIGN"
        if self.mode != old_mode:
            self.heading_pid.reset()
            self.get_logger().info(
                f"{old_mode} -> {self.mode}: heading error {math.degrees(error):+.1f} deg")

        if abs(error) <= cfg["heading_deadband"]:
            self.heading_pid.reset()
            turn = 0.0
        else:
            turn = self.heading_pid.compute(error, dt)
        forward = 0.0
        if self.mode == "DRIVE":
            approach = min(
                cfg["forward_speed"],
                cfg["curve_speed"]
            )

            # Slow down for the final destination only.
            if self.current_waypoint_idx == len(self.waypoints) - 1:
                approach = min(
                    approach,
                    cfg["distance_kp"] * tracking["distance_to_endpoint"]
                )

            heading_scale = max(
                0.0,
                1.0 - abs(error) / cfg["realign_angle"]
            )

            forward = approach * heading_scale

        return forward, turn

    def apply_motion_limits(self, forward, turn): # Not needed anymore 
        """Keep the professor's simulation workaround outside the PID math.

        Zero commands always remain zero. Real-robot minimums default to zero.
        Minimum speeds can defeat fine positioning; tune them per environment.
        """
        cfg = self.settings
        if turn != 0.0:
            turn = math.copysign(max(abs(turn), cfg["min_turn_speed"]), turn)
        if forward > 0.0:
            forward = max(forward, cfg["min_forward_speed"])
        return (float(np.clip(forward, 0.0, cfg["forward_speed"])),
                float(np.clip(turn, -cfg["turn_speed"], cfg["turn_speed"])))

    def advance_waypoint(self):

        if self.current_waypoint_idx < len(self.waypoints) - 1:
            self.current_waypoint_idx += 1

            # Restart the carrot on the next segment.
            self.carrot_from = 0.0
            self.carrot_to = self.settings["carrot_spacing"]

            return
        
        self.move_2D()
        self.current_waypoint_idx += 1
        if self.current_waypoint_idx >= len(self.waypoints):
            self.goal_idx += 1
            self.plan_new_path = self.goal_idx < len(self.goal_list)
            if not self.plan_new_path:
                self.get_logger().info("All goals reached!")
        self.reset_tracking()

    def timer_callback(self):
        """Keep optional debug failures outside the motion-control logic."""
        self._debug_tracking = None
        self._control_step()
        if self._tracking_debug is not None:
            try:
                self._tracking_debug.publish(self._debug_tracking, self._debug_command)
            except Exception as exc:
                self.get_logger().error(f"Tracking debug disabled: {exc}")
                self._tracking_debug = None

    def _control_step(self):
        """Mission checks -> tracking geometry -> motion controller -> publish."""
        if self.pose is None:
            self.get_logger().info("Waiting for pose...", throttle_duration_sec=2.0)
            return
        self._check_start_once()
        if self.path != self._last_printed_path:
            # The separate viewer displays the map when debug streaming is on.
            if self._tracking_debug is None:
                self.print_map()
            self._last_printed_path = list(self.path)
        if self.calibrate:
            self._print_calibration()
            return
        if self.goal_idx >= len(self.goal_list):
            self.move_2D()
            return
        if self.plan_new_path:
            self.plan_current_goal()
            return
        if not self.waypoints:
            self.move_2D()
            return
        if self.waypoint_reached():
            self.advance_waypoint()
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.0 if self._last_control_time is None else now - self._last_control_time
        self._last_control_time = now
        tracking = self.get_lookahead_info()
        self._debug_tracking = tracking
        forward, raw_turn = self.calculate_motion(tracking, dt)
        #forward, turn = self.apply_motion_limits(forward, raw_turn)
        turn=raw_turn 
        if self.tracking_test_only:
            self.get_logger().warn("TEST MODE")
            forward, turn = 0.0, 0.0
        self.get_logger().info(
            f"{self.mode} | goal {self.goal_idx+1}/{len(self.goal_list)} | "
            f"waypoint {self.current_waypoint_idx+1}/{len(self.waypoints)} | "
            f"carrot={self.carrot_to:.2f} m aim={tracking['use_carrot']} | "
            f"heading={math.degrees(tracking['heading_error']):+.1f} deg | "
            f"lateral={tracking['lateral_error']:+.3f} m | "
            f"endpoint={tracking['distance_to_endpoint']:.3f} m | "
            f"v={forward:.2f} raw_w={raw_turn:+.2f} w={turn:+.2f} ",
            throttle_duration_sec=1.0)
        self.move_2D(x=forward, y=0.0, turn=turn)


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
    arg_parser.add_argument('--debug-tracking', action='store_true',
                            help="Publish tracking data for the separate laptop viewer.")
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
    print(f"Map loaded: {config['map_file']}, shape={map_array.shape}, values={np.unique(map_array)}")
    waypoint = WaypointNode(map_array, config["goal_list"], is_simulation, config["origin"], config["resolution"],
                            frame=frame, robot_number=robot_number, expected_start=expected_start, calibrate=cli_args.calibrate,
                            debug_tracking=cli_args.debug_tracking)

    # Start spinning the waypoint node and only stop once SystemExit error is raised within the node callback
    try:
        rclpy.spin(waypoint)
    except SystemExit:
        print("Shutting down")

    waypoint.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()