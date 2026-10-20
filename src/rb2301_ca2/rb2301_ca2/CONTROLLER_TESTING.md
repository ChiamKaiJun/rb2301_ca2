# Carrot and PID restructure

The working copy is on `refactor/carrot-pid`. No remote branch has been pushed.
Replace both package files together: `path_planning.py` and `pid_controller.py`.
Build and source your workspace as usual before using your existing launch scripts.

## Tuning

All navigation settings are beside `CONTROL_SETTINGS` in `path_planning.py`.
`SIM_CONTROL` and `REAL_CONTROL` hold environment-specific speed limits.
The PID class receives these gains; there is no need to edit its formula to tune.

- ALIGN exits at 5 degrees; DRIVE re-enters ALIGN at 30 degrees.
- The steering deadband is independently set to 1 degree.
- Heading gains initially remain P=3.5, I=0, D=0.
- Simulation minimum commands remain 1.5 rad/s angular and 0.15 m/s forward.
- Real minimum commands remain zero. Real maximum forward speed remains 0.4 m/s.
- These are starting values, not physically validated tuning. In particular, a real
  actuator dead zone can still prevent alignment. Inspect error and raw/final turn
  output before changing a gain or minimum command.

Two independent hysteresis rules remain: 3 cm/2 cm enables/disables aiming at the
carrot, while 30 degrees/5 degrees selects ALIGN/DRIVE. Endpoints always use carrot
aiming. No strafing fallback is implemented.

## Carrot advancement

Carrots are spaced 0.4 m along each straight segment. A carrot at 0.4 m advances to
0.8 m when projected progress reaches 0.36 m. The next threshold is 0.76 m.
Progress is measured along the segment, not by distance travelled or sideways
motion. The target never passes the current corner. Corners and goals still require
both x and y errors <= 0.05 m before changing segments.

This is a discrete target, so its bearing can jump when advanced. It is not a
continuous lookahead. Start with P-only; adding D can amplify target jumps.

## Testing order

1. Set `tracking_test_only=True` for a geometry-only run. Check map/pose, segment,
   carrot and error signs. Restore False for movement tests.
2. Test a straight segment starting with 10-degree and 90-degree heading errors,
   both left and right. Confirm ALIGN exits and observe overshoot after transition.
3. Start slightly left/right of the segment. Confirm lateral correction and carrot
   advancement. Check that lateral movement alone does not advance the carrot.
4. Approach a single endpoint, then test a slightly overshot position in clear
   space. Confirm arrival is determined by position, not the 90% carrot rule.
5. Test left and right corners, then all four goals. Confirm each transition occurs
   once and the final command is zero.
6. On the real robot, first use existing calibration mode to verify the Optitrack
   coordinate transform and map, then repeat the short-segment tests at low speed.

The PID uses elapsed node-clock time. If running the node with simulated time,
ensure `use_sim_time` and the `/clock` bridge are configured together. Clock setup
was not changed by this restructure.

Run offline logic checks from the repository root:

```bash
python -m unittest discover -s tests -v
```

These checks exercise actual method definitions without ROS. They do not simulate
actuator dynamics, ROS scheduling, Gazebo contact/friction or Optitrack latency.
