"""Small, optional telemetry publisher. No plotting imports on the robot."""
import json
import math
import uuid

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String


def sample_qos():
    return QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)


def scene_qos():
    return QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


class TrackingDebug:
    def __init__(self, node):
        self.node = node
        self.run_id = uuid.uuid4().hex
        self.sequence = 0
        self.last_route = None
        self.samples = node.create_publisher(String, 'tracking/sample', sample_qos())
        self.scene = node.create_publisher(String, 'tracking/scene', scene_qos())

    def publish(self, tracking, command):
        """Called after a control tick; use its actual geometry and sent command."""
        n = self.node
        route = (tuple(n.path), tuple(n.waypoints))
        if route != self.last_route:
            # The grid uses [x, y], which the viewer transposes for imshow.
            scene = dict(run_id=self.run_id, grid=n.map_array.tolist(),
                         origin=list(n.origin), resolution=n.resolution,
                         goals=n.goal_list, path=n.path, waypoints=n.waypoints,
                         simulation=n.is_simulation)
            self.scene.publish(String(data=json.dumps(scene)))
            self.last_route = route

        if n.pose is None:
            status = 'WAIT_POSE'
        elif n.calibrate:
            status = 'CALIBRATE'
        elif n.goal_idx >= len(n.goal_list):
            status = 'DONE'
        elif not n.waypoints:
            status = 'NO_PATH'
        elif tracking is None:
            status = 'TRANSITION'
        else:
            status = n.mode
        sample = dict(run_id=self.run_id, seq=self.sequence,
                      stamp=n.get_clock().now().nanoseconds * 1e-9,
                      pose=None if n.pose is None else n.pose.tolist(),
                      mode=status, goal=min(n.goal_idx + 1, len(n.goal_list)),
                      goals=len(n.goal_list), segment=n.current_waypoint_idx,
                      segments=max(0, len(n.waypoints)-1),
                      forward=command[0], turn=command[1], source='NONE',
                      heading=None, lateral=None, target=None)
        if tracking is not None:
            sample.update(heading=math.degrees(tracking['heading_error']),
                          lateral=tracking['lateral_error'],
                          target=tracking['target'].tolist(),
                          source=tracking['heading_source'])
        self.samples.publish(String(data=json.dumps(sample, allow_nan=False)))
        self.sequence += 1
