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

class YawRatePID:
    """Track the carrot's requested yaw rate with feedforward plus PID feedback.

    Derivative acts on measured rate to avoid kicks when the carrot moves.
    Conditional integration prevents windup at the total command limit.
    """

    def __init__(self, kp=0.35, ki=0.5, kd=0.01, limit=0.4):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit = limit
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.previous_rate = None
        self.derivative = 0.0

    def update(self, desired_rate, measured_rate, dt):
        dt = float(np.clip(dt, 0.001, 0.2))
        error = desired_rate - measured_rate
        raw_derivative = (0.0 if self.previous_rate is None else
                          -(measured_rate - self.previous_rate) / dt)
        self.derivative += dt / (0.10 + dt) * (raw_derivative - self.derivative)
        self.previous_rate = measured_rate
        candidate = self.integral + error * dt
        if self.ki > 0:
            candidate = float(np.clip(candidate, -self.limit / self.ki,
                                      self.limit / self.ki))
        base = desired_rate + self.kp * error + self.kd * self.derivative
        unclipped = base + self.ki * candidate
        if abs(unclipped) <= self.limit or error * unclipped < 0:
            self.integral = candidate
        return float(np.clip(base + self.ki * self.integral,
                             -self.limit, self.limit))

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

def point_is_free(point, grid, origin, resolution, clearance=0.02):
    """Check a circular point footprint against occupied cells and map bounds.

    The assignment maps already include the robot's obstacle buffer. ``clearance``
    is only an additional tracking margin, not another robot-radius inflation.
    """
    point = np.asarray(point, dtype=float)
    origin = np.asarray(origin, dtype=float)
    upper = origin + np.asarray(grid.shape) * resolution
    if (point.shape != (2,) or not np.all(np.isfinite(point))
            or clearance < 0 or resolution <= 0):
        return False
    if np.any(point - clearance < origin) or np.any(point + clearance >= upper):
        return False
    first = np.maximum(np.floor((point - clearance - origin) / resolution).astype(int), 0)
    last = np.minimum(np.floor((point + clearance - origin) / resolution).astype(int),
                      np.asarray(grid.shape) - 1)
    for ix in range(first[0], last[0] + 1):
        for iy in range(first[1], last[1] + 1):
            if grid[ix, iy] == 0:
                continue
            lower = origin + np.array([ix, iy]) * resolution
            closest = np.clip(point, lower, lower + resolution)
            gap = point - closest
            if np.dot(gap, gap) <= clearance * clearance + 1e-12:
                return False
    return True


def segment_is_free(start, end, grid, origin, resolution, clearance=0.02):
    """Check a line at intervals of at most 1 cm, including its clearance."""
    if not all(point_is_free(p, grid, origin, resolution, clearance) for p in (start, end)):
        return False
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    length = float(np.linalg.norm(end - start))
    steps = max(1, int(math.ceil(length / min(0.01, resolution / 10))))
    # Extra margin covers the space between neighbouring checked points.
    margin = clearance + length / (2 * steps)
    return all(point_is_free(p, grid, origin, resolution, margin)
               for p in np.linspace(start, end, steps + 1))


def build_tracking_path(points, grid, origin, resolution, transition_fraction=0.8,
                        clearance=0.02, min_radius=0.04, spacing=0.02):
    """Build straight sections and checked quarter-circles from A* waypoints."""
    if (resolution <= 0 or spacing <= 0 or min_radius <= 0 or clearance < 0
            or not 0 < transition_fraction < 1):
        raise ValueError("Invalid tracking-path settings")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("Waypoints must be finite x,y pairs")
    waypoints = []
    for point in points:
        if not waypoints or np.linalg.norm(point - waypoints[-1]) > 1e-9:
            waypoints.append(point)
    if not waypoints or not point_is_free(waypoints[0], grid, origin, resolution, clearance):
        raise ValueError("Route needs a valid starting point")
    samples, curvature, distances = [waypoints[0].copy()], [0.0], [0.0]
    stops, corners = [], []

    def add_section(section, kappa=0.0):
        curvature[-1] = kappa
        for point in section[1:]:
            distances.append(distances[-1] + float(np.linalg.norm(point - samples[-1])))
            samples.append(point.copy())
            curvature.append(kappa)

    for i in range(1, len(waypoints)):
        corner = waypoints[i]
        incoming = corner - waypoints[i-1]
        # At the final goal, continue straight instead of making another turn.
        outgoing = waypoints[i+1] - corner if i+1 < len(waypoints) else incoming
        length_in, length_out = np.linalg.norm(incoming), np.linalg.norm(outgoing)
        incoming, outgoing = incoming / length_in, outgoing / length_out
        dot = float(np.dot(incoming, outgoing))
        radius = min((1-transition_fraction)*length_in, 0.45*length_in, 0.45*length_out)
        arc, kappa = None, 0.0

        # Try a quarter-circle; shrink it if it gets too close to a wall.
        while abs(dot) < 1e-6 and radius >= min_radius - 1e-9:
            entry = corner - radius * incoming
            steps = max(2, int(math.ceil(radius * math.pi / (2 * spacing))))
            candidate = [entry + radius * (math.sin(a)*incoming + (1-math.cos(a))*outgoing)
                         for a in np.linspace(0, math.pi/2, steps + 1)]
            candidate[0], candidate[-1] = entry, corner + radius * outgoing
            bulge = radius * (1 - math.cos(math.pi / (4 * steps)))
            if all(segment_is_free(a, b, grid, origin, resolution, clearance + bulge)
                   for a, b in zip(candidate, candidate[1:])):
                arc = candidate
                cross = incoming[0]*outgoing[1] - incoming[1]*outgoing[0]
                kappa = (1 if cross > 0 else -1) / radius
                break
            if radius <= min_radius + 1e-9:
                break
            radius = max(min_radius, radius * 0.75)

        # Add the straight approach to the curve, or to the original corner.
        target = corner if arc is None else arc[0]
        start = samples[-1]
        length = float(np.linalg.norm(target - start))
        if length >= 1e-9:
            if not segment_is_free(start, target, grid, origin, resolution, clearance):
                raise ValueError("Straight section is too close to an obstacle")
            steps = max(1, int(math.ceil(length / spacing)))
            add_section(np.linspace(start, target, steps + 1))

        if dot <= 1 - 1e-6:
            details = dict(point=corner.copy(), rounded=arc is not None)
            if arc is None:
                details['s'] = distances[-1]
                stops.append(distances[-1])
            else:
                details.update(radius=radius, transition_fraction=1-radius/length_in,
                               entry_s=distances[-1])
                add_section(arc, kappa)
                details['exit_s'] = distances[-1]
            corners.append(details)
    return dict(points=np.asarray(samples), s=np.asarray(distances),
                curvature=np.asarray(curvature), stops=stops, corners=corners)


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

        # Continuous carrot tracking. Only actual goals are stopping points; normal corners are driven as curves.
        self.corner_transition = 0.80
        self.lookahead_distance = 0.30 if is_simulation else 0.20
        self.minimum_lookahead = 0.04
        # The supplied grid already includes a robot buffer. This is an EXTRA
        # tracking allowance, not a replacement for the robot's footprint.
        self.path_clearance = 0.02
        self.reference_clearance = 0.05  # Leave 3 cm for tracking error before the motion guard.
        self.minimum_curve_radius = 0.04
        self.tracking_test_only = False
        self.mode = "ALIGN"
        self.forward_speed = 0.5 if self.is_simulation else 0.20
        self.turn_speed = 2.8 if self.is_simulation else 0.4
        # Retain the existing sim breakaway workaround ONLY for stationary alignment. 
        # A minimum turn command during DRIVE would ruin curves.
        self.min_alignment_turn = 1.5 if self.is_simulation else 0.0
        self.heading_tolerance = np.deg2rad(3.0)
        self.heading_kp = 5.0 if self.is_simulation else 2.0
        self.distance_kp = 2.0
        self.realign_angle = np.deg2rad(100.0)
        self.goal_tolerance = 0.1  # Accept actual goals within 5 cm.        
        self.corner_stop_tolerance = 0.02
        self.acceleration = 0.4 if self.is_simulation else 0.15
        self.rate_pid = YawRatePID(limit=self.turn_speed)
        self.tracking_path = None
        self.route_progress = 0.0
        self.stop_index = 0
        self.finished = False
        self.measured_yaw_rate = 0.0
        self._previous_pose_sample = None
        self._sample_uses_sensor_time = None
        self._control_stamp = None
        self._pose_received_at = None
        self._pose_sequence = 0
        self._controlled_sequence = -1
        self._last_forward = 0.0
        self._last_command = (0.0, 0.0)

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

        grid.print_grid_map(
            waypoints=goal_cells,
            path=self.path,
        )

        grid.draw_grid_map(
            waypoints=self.waypoints,
            path=self.path,
            goals=goal_cells,
        )

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
        pose = np.array((x,y,heading))
        if self._update_motion_feedback(msg, pose):
            self.pose = pose
        return self.pose

    def odometer_callback(self, msg):
        '''Callback to calculate 2D pose info from Gazebo odomoter. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        latest_pose_msg = msg.pose.pose
        heading = np.rad2deg(self.yaw_from_quaternion(latest_pose_msg.orientation))
        pose = np.array((latest_pose_msg.position.x, latest_pose_msg.position.y, heading))
        if self._update_motion_feedback(msg, pose, msg.twist.twist.angular.z):
            self.pose = pose
        return self.pose

    def _update_motion_feedback(self, msg, pose, yaw_rate=None):
        """Use sensor time for rate estimates; receipt time detects lost poses."""
        received = time.monotonic()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if not np.all(np.isfinite(pose)) or (yaw_rate is not None and not math.isfinite(yaw_rate)):
            return False
        # Zero is a valid first Gazebo timestamp. Some real pose sources use
        # zero to mean unstamped; never compare wall time with sensor time.
        uses_sensor_time = self.is_simulation or stamp > 0
        stamp = stamp if uses_sensor_time else received
        if uses_sensor_time != self._sample_uses_sensor_time:
            self._previous_pose_sample = None
            self._control_stamp = None
            self.rate_pid.reset()
        self._sample_uses_sensor_time = uses_sensor_time
        yaw = float(np.deg2rad(pose[2]))
        if self._previous_pose_sample is not None:
            previous_stamp, previous_yaw = self._previous_pose_sample
            dt = stamp - previous_stamp
            if 0.001 <= dt <= 0.5:
                raw_rate = (wrap_angle(yaw - previous_yaw) / dt
                            if yaw_rate is None else yaw_rate)
                alpha = dt / (0.08 + dt)
                self.measured_yaw_rate += alpha * (raw_rate - self.measured_yaw_rate)
            elif dt <= 0:
                return False  # Repeated/out-of-order timestamp is not fresh feedback.
            else:
                self.rate_pid.reset()
                self.measured_yaw_rate = 0.0 if yaw_rate is None else float(yaw_rate)
        self._previous_pose_sample = (stamp, yaw)
        self._pose_received_at = received
        self._pose_sequence += 1
        return True

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
    def stop_motion(self):
        """Clear controller memory when stopping, never at ordinary corners."""
        self.rate_pid.reset()
        self._last_forward = 0.0
        self._last_command = (0.0, 0.0)
        self.move_2D()

    def plan_current_goal(self):
        """Keep A* for planning, then replace sharp corners with checked arcs."""
        self.plan_new_path = False
        self.stop_motion()
        self.tracking_path = None
        self.path, self.waypoints = [], []
        start = world_to_grid(*self.pose[:2], self.origin, self.resolution)
        goal = world_to_grid(*self.goal_list[self.goal_idx], self.origin, self.resolution)
        route = astar(self.map_array, start, goal)
        if route is None:
            self.get_logger().error(f"No route found: {start} -> {goal}")
            return
        self.path = route
        self.waypoints = get_turn_waypoints(route)
        points = [grid_to_world(*cell, self.origin, self.resolution)
                  for cell in self.waypoints]
        # Visit the actual requested coordinate, including non-centred goals.
        if len(points) == 1:
            points = [tuple(self.pose[:2]), self.goal_list[self.goal_idx]]
        else:
            points[-1] = self.goal_list[self.goal_idx]
        try:
            self.tracking_path = build_tracking_path(
                points, self.map_array, self.origin, self.resolution,
                transition_fraction=self.corner_transition,
                clearance=self.reference_clearance, min_radius=self.minimum_curve_radius)
        except ValueError as error:
            self.get_logger().error(f"Cannot build a clear tracking route: {error}")
            return
        self.route_progress = 0.0
        self._carrot_s = 0.0
        self._lookahead_length = self.lookahead_distance
        self.stop_index = 0
        self.mode = "ALIGN"
        self._controlled_sequence = -1
        self._control_stamp = None
        # Speed depends on curvature/braking distance, never distance to carrot.
        reference = self.tracking_path
        curvature = np.abs(reference['curvature'])
        speed = np.minimum(self.forward_speed,
                           0.8 * self.turn_speed / np.maximum(curvature, 1e-9))
        speed[-1] = 0.0
        for stop_s in reference['stops']:
            speed[np.argmin(abs(reference['s'] - stop_s))] = 0.0
        for i in range(len(speed) - 2, -1, -1):
            ds = reference['s'][i + 1] - reference['s'][i]
            speed[i] = min(speed[i], math.sqrt(speed[i + 1]**2 + 2*self.acceleration*ds))
        reference['speed'] = speed
        rounded = sum(corner['rounded'] for corner in reference['corners'])
        self.get_logger().info(
            f"Goal {self.goal_idx + 1}: {len(route)-1} grid steps, "
            f"{rounded} rounded corners, {len(reference['stops'])} stop-and-turn corners. "
            f"Preferred turn transition: {self.corner_transition:.0%}.")

    def reference_point(self, distance):
        reference = self.tracking_path
        return np.array([np.interp(distance, reference['s'], reference['points'][:, axis])
                         for axis in range(2)])

    def get_lookahead_info(self, dt):
        """Project locally in route order, then interpolate a carrot across curves."""
        reference = self.tracking_path
        points, distances = reference['points'], reference['s']
        end_s = (reference['stops'][self.stop_index]
                 if self.stop_index < len(reference['stops']) else distances[-1])
        # A small search window prevents jumping to a nearby later corridor.
        low = max(0.0, self.route_progress - 0.05)
        high = min(end_s, self.route_progress + max(0.08, 2*self.forward_speed*dt))
        best_distance, best_s = float('inf'), self.route_progress
        for i in range(max(0, np.searchsorted(distances, low) - 1),
                       min(len(points)-1, np.searchsorted(distances, high) + 1)):
            a, b = points[i], points[i+1]
            segment = b-a
            length_squared = float(np.dot(segment, segment))
            fraction = float(np.clip(np.dot(self.pose[:2]-a, segment) / length_squared,
                                     0.0, 1.0))
            candidate_s = float(np.clip(distances[i] + fraction*math.sqrt(length_squared), low, high))
            candidate = self.reference_point(candidate_s)
            error = float(np.linalg.norm(candidate-self.pose[:2]))
            if error < best_distance:
                best_distance, best_s = error, candidate_s
        self.route_progress = max(self.route_progress, best_s)
        # Short carrots on tight arcs limit corner cutting. Reduce lookahead
        # continuously on approach, rather than jumping when a curve appears.
        lookahead = self.lookahead_distance
        for corner in reference['corners']:
            if (corner['rounded'] and corner['exit_s'] >= self.route_progress
                    and corner['entry_s'] < end_s):
                approach = max(0.0, corner['entry_s']-self.route_progress)
                lookahead = min(lookahead, max(self.minimum_lookahead,
                                               0.6*corner['radius'] + 0.5*approach))
        self._lookahead_length += float(np.clip(lookahead-self._lookahead_length, -0.5*dt, 0.5*dt))
        target_s = min(end_s, max(self._carrot_s, self.route_progress+self._lookahead_length))
        self._carrot_s = target_s
        target = self.reference_point(target_s)
        delta = target - self.pose[:2]
        distance = float(np.linalg.norm(delta))
        heading = float(np.deg2rad(self.pose[2]))
        lateral_body = -math.sin(heading)*delta[0] + math.cos(heading)*delta[1]
        curvature = 2*lateral_body / max(distance*distance, 1e-9)
        heading_error = float(wrap_angle(math.atan2(delta[1], delta[0])-heading))
        endpoint = self.reference_point(end_s)
        return {'target': target, 'target_s': target_s, 'end_s': end_s,
                'distance_to_lookahead': distance, 'heading_error': heading_error,
                'curvature': float(curvature),
                'remaining': max(0.0, end_s-self.route_progress),
                'endpoint_distance': float(np.linalg.norm(endpoint-self.pose[:2])),
                'lateral_error': float(np.linalg.norm(self.reference_point(self.route_progress)-self.pose[:2]))}

    def command_is_clear(self, forward, turn):
        """Check the commanded arc over reaction plus braking time in the buffered map."""
        position = self.pose[:2].copy()
        heading = float(np.deg2rad(self.pose[2]))
        horizon = 0.20 + abs(forward) / self.acceleration
        count = max(2, int(math.ceil(horizon / 0.05)))
        dt = horizon / count
        for _ in range(count):
            if abs(turn) < 1e-9:
                following = position + forward*dt*np.array([math.cos(heading), math.sin(heading)])
            else:
                following = position + forward/turn*np.array([
                    math.sin(heading+turn*dt)-math.sin(heading),
                    math.cos(heading)-math.cos(heading+turn*dt)])
            sagitta = (abs(forward/turn)*(1-math.cos(turn*dt/2)) if abs(turn) > 1e-9 else 0.0)
            if not segment_is_free(position, following, self.map_array, self.origin,
                                   self.resolution, self.path_clearance+sagitta):
                return False
            position, heading = following, heading+turn*dt
        return True

    def calculate_motion(self, tracking, dt):
        heading_error = tracking['heading_error']
        if self.mode == 'ALIGN' and abs(heading_error) <= self.heading_tolerance:
            self.mode = 'DRIVE'
            self.rate_pid.reset()
        elif self.mode == 'DRIVE' and abs(heading_error) >= self.realign_angle:
            self.mode = 'ALIGN'  # Recovery only, not a normal 90-degree corner.
            self.rate_pid.reset()
            self._last_forward = 0.0
        if self.mode == 'ALIGN':
            desired_rate = float(np.clip(self.heading_kp*heading_error,
                                         -self.turn_speed, self.turn_speed))
            turn = self.rate_pid.update(desired_rate, self.measured_yaw_rate, dt)
            if self.min_alignment_turn and abs(heading_error) > self.heading_tolerance:
                turn = float(np.copysign(max(abs(turn), self.min_alignment_turn), heading_error))
            return 0.0, float(np.clip(turn, -self.turn_speed, self.turn_speed))

        reference = self.tracking_path
        # Evaluate slightly ahead so acceleration can resume after a fallback stop.
        profile_s = min(tracking['end_s'], self.route_progress + 0.01)
        forward = float(np.interp(profile_s, reference['s'], reference['speed']))
        forward = min(forward, self.forward_speed,
                      0.8*self.turn_speed / max(abs(tracking['curvature']), 1e-9))
        # Only a true goal or explicitly unrounded corner causes endpoint braking.
        forward = min(forward, self.distance_kp*tracking['endpoint_distance'],
                      math.sqrt(2*self.acceleration*tracking['endpoint_distance']))
        forward = min(forward, self._last_forward + self.acceleration*dt)
        # If projection reaches the endpoint before the robot, still approach
        # its actual position instead of getting stuck at a zero speed sample.
        if tracking['remaining'] < 0.01:
            forward = min(self.forward_speed, self.distance_kp*tracking['endpoint_distance'],
                          0.8*self.turn_speed / max(abs(tracking['curvature']), 1e-9),
                          self._last_forward + self.acceleration*dt)
        desired_rate = forward * tracking['curvature']
        turn = self.rate_pid.update(desired_rate, self.measured_yaw_rate, dt)
        # Predict the actual PID command too, not just the ideal reference curve.
        # Scaling both components preserves the commanded radius.
        for scale in (1.0, 0.75, 0.5, 0.25):
            if self.command_is_clear(forward*scale, turn*scale):
                if scale < 1.0:
                    self.rate_pid.reset()
                self._last_forward = forward*scale
                return float(forward*scale), float(turn*scale)
        self.mode = 'BLOCKED'
        self.get_logger().error('Tracking command approaches an occupied cell; stopped. Check pose/map alignment.')
        self.stop_motion()
        return 0.0, 0.0

    def advance_goal(self):
        self.stop_motion()
        self.get_logger().info(f"Goal {self.goal_idx + 1} reached within {self.goal_tolerance:.2f} m.")
        if self.goal_idx + 1 == len(self.goal_list):
            self.finished = True
            self.get_logger().info('All goals reached!')
        else:
            self.goal_idx += 1
            self.plan_new_path = True

    def timer_callback(self):
        if self.pose is None:
            return
        if self.calibrate:
            self._print_calibration()
            return
        self._check_start_once()
        if self.plan_new_path:
            self.plan_current_goal()
        if self.path != self._last_printed_path:
            self.print_map()
            self._last_printed_path = list(self.path)
        if self.finished or self.tracking_path is None or self.mode == 'BLOCKED':
            self.stop_motion()
            return
        # Never continue on an old Optitrack pose. Simulation gets a longer
        # allowance because odometry may pause while stationary / Gazebo loads.
        timeout = 2.0 if self.is_simulation else 0.5
        if self._pose_received_at is None or time.monotonic()-self._pose_received_at > timeout:
            self.stop_motion()
            self.get_logger().warn('Waiting for fresh pose feedback.', throttle_duration_sec=1.0)
            return
        if self._controlled_sequence == self._pose_sequence:
            self.move_2D(x=self._last_command[0], y=0.0, turn=self._last_command[1])
            return  # Do not integrate PID again on the same sensor reading.
        stamp = self._previous_pose_sample[0]
        dt = 0.05 if self._control_stamp is None else float(np.clip(stamp-self._control_stamp, 0.001, 0.2))
        self._control_stamp = stamp
        self._controlled_sequence = self._pose_sequence
        if np.linalg.norm(self.pose[:2]-np.asarray(self.goal_list[self.goal_idx])) <= self.goal_tolerance:
            self.advance_goal()
            return
        tracking = self.get_lookahead_info(dt)
        if (self.stop_index < len(self.tracking_path['stops'])
                and tracking['remaining'] <= self.corner_stop_tolerance
                and tracking['endpoint_distance'] <= self.corner_stop_tolerance):
            self.route_progress = tracking['end_s']
            self.stop_index += 1
            self.mode = 'ALIGN'
            self.stop_motion()
            return
        if tracking['distance_to_lookahead'] <= 1e-8:
            self.stop_motion()
            return
        if self.tracking_test_only:
            self.stop_motion()
            return
        forward, turn = self.calculate_motion(tracking, dt)
        self._last_command = (forward, turn)
        self.move_2D(x=forward, y=0.0, turn=turn)
        self.get_logger().info(
            f"{self.mode} | GOAL {self.goal_idx+1}/{len(self.goal_list)} | "
            f"PATH {self.route_progress:.2f}/{self.tracking_path['s'][-1]:.2f} m | "
            f"CARROT ({tracking['target'][0]:.2f}, {tracking['target'][1]:.2f}) | "
            f"LATERAL {tracking['lateral_error']:.3f} m | "
            f"FORWARD {forward:.3f} m/s | TURN {turn:+.3f} rad/s | "
            f"MEASURED TURN {self.measured_yaw_rate:+.3f} rad/s",
            throttle_duration_sec=1.0)


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
        '''Builds the colour image: blue walls, white free space, green path and pink waypoints.'''
        image_grid = np.ones((self.grid.shape[0],self.grid.shape[1],3), dtype=np.uint8)
        image_grid[self.grid <= obstacle_threshold] = (255,255,255)
        image_grid[self.grid > obstacle_threshold] = (0,0,255)

        for x, y in path:
            image_grid[x][y] = (0,255,0)

        for point in waypoints:
            image_grid[point] = (255,105,180)

        image_grid[self.starting_position] = (255,255,0)  # Yellow start

        return image_grid

    def draw_grid_map(self, waypoints:list=(), path:list=(), obstacle_threshold:float=50, save_path:str=None, show:bool=True, goals:list=(),):
        '''Creates an image with blue walls, white free space, green path, pink waypoints and red goals.

        Args:
            waypoints : grid cells drawn in pink, taking precedence over the path
            path : list (or other iterable) of tuple coordinates representing all the grid indices forming the solution path. Will be represented in green
            obstacle_threshold : Optional float to indicate threshhold for whether a grid is considered occupied. Not important for ca2
            save_path : Optional file path (e.g. "path_result.png") to save the image to, in addition to/instead of showing it
            show : Whether to pop up the image in a viewer (default True). Set to False if you only want to save it
            goals : assignment goal cells drawn in red, taking precedence over the path and waypoints
        '''
        image_grid = self._colour_grid(waypoints, path, obstacle_threshold)

        # Paint every assignment goal red.
        for goal in goals:
            image_grid[goal] = (255, 0, 0)

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

    try:
        rclpy.spin(waypoint)
    except (KeyboardInterrupt, SystemExit):
        print("Shutting down")
    finally:
        # Also stop if a callback raises, while the ROS context is still usable.
        if rclpy.ok() and not waypoint.calibrate:
            waypoint.stop_motion()
        waypoint.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
