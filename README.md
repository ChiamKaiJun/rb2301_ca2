# RB2301 CA2 - Robot Path Planning

The robot has to drive through a known maze and reach a list of goal points in order, without touching a wall. There are two parts to it: plan a path on the occupancy grid (A*, Dijkstra, BFS, your choice), then make the robot follow that path using waypoints and PID control.

You write and test everything in the Gazebo simulation first (sections 1 to 3). Then you run the same file on the real robot in the lab maze (section 4). Nothing in your code changes between the two.

Placeholders in `<angle brackets>` or capitals (`NN`, `<group>`) are things you replace with your own values.

---

## 1. Where your code goes

Everything you write goes in one file:

```
src/rb2301_ca2/rb2301_ca2/path_planning.py
```

- Your path planning and PID code goes in `WaypointNode.timer_callback()`, between the two `###### INSERT CODE HERE ######` lines. It runs 20 times a second.
- Variables that have to survive between calls (current goal, waypoint list, PID error terms) go in `WaypointNode.__init__()`.
- Add your own helper functions to the same file if you want.

Leave the other files alone (launch files, world, maps, scripts, config).

### What you get inside `WaypointNode`

| Name | What it is |
|---|---|
| `self.pose` | Robot pose `[x, y, heading]`. Metres, metres, degrees (-180 to 180). Heading 0 is along +x and counter-clockwise is positive. It is `None` until the first reading comes in. Same meaning in the simulation and on the real robot. |
| `self.goal_list` | The goals to visit, in order, as `(x, y)`. |
| `self.map_array` | The occupancy grid, a numpy array. `0` is free, `99` is wall. |
| `self.origin`, `self.resolution` | Where the grid sits in the world and the size of one cell. |
| `self.move_2D(x, y, turn)` | Drive the robot: forward speed `x`, sideways speed `y`, turning speed `turn`. Values are clipped to the maximum speed. |
| `self.path` | Put your planned route here (list of grid cells). It is drawn in the terminal for you, see below. |
| `grid_to_world(i, j, origin, resolution)` | Grid cell to world `(x, y)` of the centre of that cell. |
| `world_to_grid(x, y, origin, resolution)` | World `(x, y)` to the cell `(i, j)` that contains it. |
| `Grid` | Helper class: `check_grid_validity()` (is a route possible at all), `draw_grid_map()`, `print_grid_map()`, `animate_path()`. |

Call the two conversion functions with `self.origin` and `self.resolution`, for example `world_to_grid(x, y, self.origin, self.resolution)`.

### How the grid maps to the world

- The grid is indexed `[i, j]`. `i` goes up with world x, `j` goes up with world y.
- In the simulation a cell is 0.2 m by 0.2 m and the corner of cell `[0, 0]` is at world `(-1.0, -5.0)`.
- A cell's position is its centre: cell `[0, 0]` is at `(-0.9, -4.9)`, cell `[1, 0]` at `(-0.7, -4.9)`. The conversion functions already do this. Use them instead of working it out yourself.
- The walls in the map already include a safety margin, so any path through free cells is safe.

### Simulation goals

The robot starts at `(0.0, 0.0)` and the goals, in order, are:

```
(3.5, -3.5)  ->  (3.3, 0.3)  ->  (2.5, -3.5)  ->  (-0.3, -3.7)
```

The real maze has its own goals, listed in section 4.

### The map in your terminal

When you run the node, it prints the map as soon as the first pose arrives:

```
#  wall        .  free cell        S  the robot's current cell
W  goal point  G  final goal       *  your planned route
```

Set `self.path = [(i1, j1), (i2, j2), ...]` (grid cells in travel order) and the map is printed again with your route on it. Handy for checking a planner before you let the robot drive. If you want a picture instead, `Grid(...).draw_grid_map(path=..., waypoints=..., save_path="route.png")` works too.

### Movement

- `move_2D` takes a sideways speed `y` because the robot can move sideways.
- Bonus: reach the goals using only forward/backward motion and turning, no sideways speed, like a normal car.
- Speed limit, applied inside `move_2D`: 1.4 m/s in the simulation and 0.4 m/s on the real robot. Don't change it.

---

## 2. Set up your laptop

You need Ubuntu with ROS 2 Jazzy and Gazebo Harmonic, the bridge package `ros-jazzy-ros-gz`, the Python packages `numpy` and `pillow`, and your existing RB2301 workspace (the folder that has `src/` in it, for example `~/Documents/rb2301`).

1. Download this repo: `git clone https://github.com/rudra-8000/rb2301_ca2.git`, or press the green Code button and choose Download ZIP.
2. Copy the folders `rb2301_ca2` and `rb2301_gz` from its `src/` into your workspace's `src/`. If your workspace already has an `rb2301_gz` from CA1, replace it, the new one has the CA2 world in it.
3. Copy `ca2.sh` and `gz_ca2.sh` into the workspace folder itself, next to `src/`, and make them executable.

If the repo is in `~/rb2301_ca2` and your workspace is `~/Documents/rb2301`, that is:

```bash
cd ~/Documents/rb2301
rm -rf src/rb2301_gz src/rb2301_ca2        # only if they are already there
cp -r ~/rb2301_ca2/src/rb2301_ca2 ~/rb2301_ca2/src/rb2301_gz src/
cp ~/rb2301_ca2/ca2.sh ~/rb2301_ca2/gz_ca2.sh .
chmod +x ca2.sh gz_ca2.sh
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
```

The build should finish without errors (the last line starts with `Summary:`). After that, edits to `path_planning.py` take effect straight away, you only rebuild if you add or remove files.

### Already have the earlier simulation version?

You can keep your solution. Only `path_planning.py` is different in a way that matters: it now also supports the real robot (Optitrack pose, the real maze, `--run`, `--calibrate`). The parts you wrote stay exactly as they are: your variables in `__init__()`, your code between the two `INSERT CODE HERE` lines, your helper functions. The simulation behaves as before.

1. Back up your file: `cp src/rb2301_ca2/rb2301_ca2/path_planning.py ~/path_planning_mine.py`
2. Replace everything else with the new files, same as the steps above: `src/rb2301_ca2` (except `path_planning.py`), `src/rb2301_gz` and the two scripts.
3. Paste your solution into the new `path_planning.py`: your `__init__` variables, the code between the markers and your helpers. To see what you added, compare the two files, for example `diff -u ~/path_planning_mine.py src/rb2301_ca2/rb2301_ca2/path_planning.py`, or use the compare feature in VS Code.
4. Run `colcon build --symlink-install` and try the simulation again.

If you use git: commit your work first, then `git pull`. Git keeps your changes and brings in the new ones. If it reports a conflict in `path_planning.py`, open the file, look for the `<<<<<<<` and `>>>>>>>` blocks, keep both your lines and the new ones, then `git add` and `git commit`. If you would rather not deal with that, take the other files with `git checkout origin/main -- src/rb2301_gz src/rb2301_ca2/rb2301_ca2/optitrack_variables.config src/rb2301_ca2/rb2301_ca2/ca2_irl_map.npy src/rb2301_ca2/rb2301_ca2/ca2_irl_layout.json` and merge `path_planning.py` by hand.

---

## 3. Run the simulation

Open two terminals, both in your workspace folder.

```bash
./gz_ca2.sh      # terminal 1
./ca2.sh         # terminal 2, after the Gazebo window is open and the robot is in the maze
```

The robot starts at `(0, 0)`. Once the node gets its first pose it prints the map. Stop either terminal with `Ctrl+C`. Stopping Gazebo also kills any leftover Gazebo processes. To start the robot somewhere else, change `x:=` and `y:=` in `gz_ca2.sh`.

Problems:

- `Permission denied` on a `.sh` file: run `chmod +x gz_ca2.sh ca2.sh`.
- Gazebo hangs on "Requesting list of world names": an old Gazebo is still running. Run `pkill -9 ruby` and start `./gz_ca2.sh` again.
- Nothing happens when you run `./ca2.sh`: Gazebo has to be fully loaded first. Check terminal 1 is still running and try again.
- `ros2: command not found` or package not found: run `source /opt/ros/jazzy/setup.bash`, and check that `colcon build` finished without errors. The scripts need to be run from the workspace folder (the one that has `install/` in it).
- Gazebo window is black, very slow, or crashes (virtual machine, no GPU): in `src/rb2301_gz/launch/ca2_gazebo.launch.py` change `SetEnvironmentVariable("LIBGL_ALWAYS_SOFTWARE", "0")` to `"1"` and start the simulation again.

---

## 4. Run on the real robot

### The maze

The maze is 3.2 m by 3.2 m on the same 0.2 m grid as the simulation (`ca2_irl_map.npy`, 16 x 16 cells, `0` free and `99` wall). Coordinates are in metres with `(0, 0)` at the outer corner of cell `[0, 0]`, x along the first array axis and y along the second. There are three runs, chosen with `--run`:

| `--run` | Robot starts at | Goals, in order |
|---|---|---|
| `test1` | (0.5, 0.5) | (2.9, 1.1) -> (1.7, 0.5) -> (2.3, 0.5) -> (1.7, 1.1) |
| `test2` | (2.7, 2.7) | (0.5, 1.1) -> (1.7, 2.3) -> (0.5, 2.9) -> (1.1, 2.9) |
| `full`  | (0.5, 2.1) | (1.1, 0.5) -> (2.3, 2.9) -> (2.9, 0.5) -> (0.5, 1.7) |

Put the robot on the start point of your run before you launch. If it is more than 0.4 m away you get a warning.

Your code stays the same as in the simulation. The Optitrack pose is converted to the maze's own coordinates before your code sees it, so `self.origin` is `(0, 0)`, `self.resolution` is `0.2`, and `world_to_grid()` and `grid_to_world()` behave as they do in the simulation.

`optitrack_variables.config` holds the calibration that maps the real maze in Optitrack onto the maze in your code (the grid / numpy array). It is already filled in. Don't edit it.

### Robot numbers

Robot `NN` (two digits, so robot 7 is `07`) lives at `bingda@192.168.1.2NN`, and its Optitrack rigid body and ROS topic are called `bingda_0NN`. `ROS_DOMAIN_ID` is already set to the robot's number on every robot, and the node picks its robot from that. You don't pass or configure anything. The `vrpn` client publishes every rigid body and your node only listens to `/vrpn_mocap/bingda_0NN/pose`, the one for the robot it is running on.

One rule about the rigid body: it has to be created in Motive with the robot facing +y, which is up the left side of the printed map (from (0.5, 0.5) towards (0.5, 2.1)). The code then reports heading `+90` when the robot faces +y and `0` when it faces +x. If you ever reset a rigid body, point the robot at +y first.

### Steps

You edit on your laptop and copy the package to the robot. The robot runs Ubuntu 20.04 with ROS 2 Foxy. Only `src/rb2301_ca2` goes to the robot, so there you start the node with `ros2 run` and not `ca2.sh`.

**1. Make a workspace on the robot** (ssh in as usual). Use your group number so groups don't overwrite each other:

```bash
mkdir -p ~/Downloads/rb2301_ca2_11/src
```

**2. Copy your package.** Run this on your laptop, in a normal terminal, not in the ssh one. Repeat it every time you change `path_planning.py`:

```bash
rsync -auvx --delete ~/Documents/rb2301/src/rb2301_ca2 bingda@192.168.1.2NN:~/Downloads/rb2301_ca2_<group>/src
```

The first path is your own `src/rb2301_ca2` on the laptop. `--delete` only affects that one folder on the robot.

**3. Build on the robot** (ssh terminal). Once is enough unless you add or remove files:

```bash
cd ~/Downloads/rb2301_ca2_<group>
colb
```

`colb` is an alias for `colcon build --symlink-install`. It should end with `Summary: 1 package finished`.

**4. Start three things, each in its own ssh terminal on the robot:**

```bash
basecontrol          # terminal 1, drives the wheels
vrpn                 # terminal 2, Optitrack poses
cd ~/Downloads/rb2301_ca2_<group> && source install/setup.bash && ros2 run rb2301_ca2 path_planning --run test1     # terminal 3
```

After `vrpn` is up, `ros2 topic list` should show `/vrpn_mocap/bingda_0NN/pose`. Start terminal 3 last, with the robot on the start point of the run you chose (`--run test2` and `--run full` are the other two). It prints which robot it found (`robot=bingda_0NN`), then the map with the robot `S`, the goals `W` and `G` and later your route `*`.

If you changed nothing in `path_planning.py` yet, the starter code just drives the robot straight ahead as soon as it has a pose. Start with a clear lane in front of it.

Stay out of the Optitrack area while the robot runs. People and objects in it can disturb the tracking. To stop the robot press `Ctrl+C` in the `path_planning` terminal. `Ctrl+C` in the `basecontrol` terminal stops the wheels too.

### Checking the pose before you drive

Add `--calibrate` to the command in terminal 3. The node then prints the raw Optitrack pose and the maze pose twice a second and never sends a drive command, so you can carry the robot around by hand:

```bash
ros2 run rb2301_ca2 path_planning --run test1 --calibrate
```

With the robot on the start point the maze pose should match it within a few cm and the cell number should be right. Facing +x the heading should be about 0. If it isn't, tell a TA.

### Aliases already on the robot

| Alias | Does |
|---|---|
| `basecontrol` | `ros2 launch base_control_ros2 base_control.launch.py`, the wheel driver |
| `vrpn` | `ros2 launch vrpn_mocap client.launch.yaml server:=192.168.1.199 port:=3883`, the Optitrack poses |
| `colb` | `colcon build --symlink-install` |
| `keyboard` | `ros2 run teleop_keyboard keyboard`, drive by hand (useful to get onto the start point) |
| `rplidar` | `ros2 launch rplidar_ros rplidar.launch.py`, the lidar, not needed here |
| `talker`, `listener` | ROS 2 demo nodes, for testing the network |
| `nanobash`, `sourcebash` | `sudo nano ~/.bashrc` to edit the file, `source ~/.bashrc` to reload it |
| `unsetpath` | clears `AMENT_PREFIX_PATH` and `CMAKE_PREFIX_PATH`, for when a build picks up the wrong workspace |

To add your own, run `nanobash`, put a new line at the bottom, save, then run `sourcebash` (or open a new terminal). For example:

```bash
alias ca2='cd ~/Downloads/rb2301_ca2_<group> && source install/setup.bash && ros2 run rb2301_ca2 path_planning --run test1'
```

Don't touch the `source` lines or `export ROS_DOMAIN_ID=...` in that file. Everyone using the robot shares it, so give your alias a name nobody else will pick and leave other people's lines alone.

On your laptop you can do the same in `~/.bashrc` to shorten the copy command:

```bash
alias ca2sync='rsync -auvx --delete ~/Documents/rb2301/src/rb2301_ca2 bingda@192.168.1.2NN:~/Downloads/rb2301_ca2_<group>/src'
```

### If something goes wrong

- Nothing prints and the robot doesn't move: the node waits for a pose. Is `vrpn` running, and does `ros2 topic list` show `/vrpn_mocap/bingda_0NN/pose`? Is `basecontrol` running?
- The node runs but the robot doesn't move: check that `basecontrol` is still running in terminal 1.
- `Robot is at ... m away`: the robot isn't on the start point. If it is, ask a TA.
- The robot shows up in the wrong cell, drives into walls, or turns the wrong way: ask a TA. The mapping or the rigid body needs a look. Don't edit the config yourself.
- Build errors, or your change isn't showing up: rsync again, then run `colb` on the robot.

---

## What is in this repo

```
ca2.sh, gz_ca2.sh              copy next to src/ in your workspace (simulation)
src/rb2301_ca2/                copy into your workspace's src/, and rsync to the robot
  rb2301_ca2/path_planning.py  your solution goes here
  rb2301_ca2/ca2_sim_map.npy   simulation maze grid
  rb2301_ca2/ca2_irl_map.npy   real maze grid
  rb2301_ca2/ca2_irl_layout.json   start and goals of the three real runs
  rb2301_ca2/optitrack_variables.config   real maze placement in Optitrack, don't edit
src/rb2301_gz/                 copy into your workspace's src/ (Gazebo world, robot model)
tools/irl_calibrate.py         for TAs, see below
```

### For TAs

Calibration is done once per maze placement, and again if the maze or Optitrack gets moved. Put a robot on two or more known maze points far apart and read the raw pose with `--calibrate`. Then run `python3 tools/irl_calibrate.py --pair 0.5 0.5 <opti_x> <opti_y> --pair 2.7 2.7 <opti_x> <opti_y>`. The residual should be 1 to 2 cm and it warns above 5 cm. Paste `origin_x`, `origin_y` and `rotation_deg` into `[frame]` in `optitrack_variables.config`. Motive has to stream Z-up.

Rigid bodies are created facing maze +y, which is what `heading_offset_deg = auto` assumes. A robot whose rigid body was made differently gets its own line under `[heading_offsets]` as `<robot number> = <reported heading minus true maze heading>` (robots 7 and 12 are measured). The real-robot speed limit is `max_translate_velocity` in `load_irl_config()` in `path_planning.py`.


## Our Shortcuts

ssh bingda@192.168.1.205

mkdir -p ~/Downloads/rb2301_ca2_11/src

rsync -auvx --delete ~/rb2301_ca2/src/rb2301_ca2 bingda@192.168.1.2NN:~/Downloads/rb2301_ca2_11/src


cd ~/Downloads/rb2301_ca2_11

colb

echo $ROS_DOMAIN_ID

3 terminals 

basecontrol          # terminal 1, drives the wheels

vrpn                 # terminal 2, Optitrack poses

cd ~/Downloads/rb2301_ca2_<group> && source install/setup.bash && ros2 run rb2301_ca2 path_planning --run test1 --debug-tracking    # terminal 3


ros2 topic list # check for vrpn

### ALIAS
~/.bashrc

alias ca2='cd ~/Downloads/rb2301_ca2_11 && source install/setup.bash && ros2 run rb2301_ca2 path_planning --run test1'

alias ca2sync='rsync -auvx --delete ~/rb2301/src/rb2301_ca2 bingda@192.168.1.2NN:~/Downloads/rb2301_ca2_11/src'

Ctrl + O # Save 

Ctrl + X # Exit

source ~/.bashrc


### Plotting (LAPTOP terminal)

sudo apt install python3-matplotlib python3-tk

source install/setup.bash

export ROS_DOMAIN_ID=5 # replace with robot ID

ros2 run rb2301_ca2 tracking plotter

ros2 run rb2301_ca2 tracking_plotter \
  --map-rotation 90 \
  --csv tracking_logs/test2_low_speed.csv

## Tuning

| Test / observation | What to adjust first |
|---|---|
| Turn on the spot: heading repeatedly overshoots zero while the heading reference stays steady | Reduce heading_kp slightly. |
| Heading converges too slowly, and turn command is below its limit | Increase heading_kp slightly. |
| Heading converges slowly, but turn command stays at its limit | Increasing Kp won’t help while saturated. Check REAL_CONTROL["turn_speed"] and actual robot response. |
| Long straight with a lateral offset: correction is too slow | Reduce carrot_spacing slightly. A closer carrot generally demands a stronger correction. |
| Long straight: robot weaves left/right while moving | Reduce REAL_CONTROL["forward_speed"] first. If needed, try a larger carrot_spacing. |
| Short consecutive corners: lateral error carries into the next segment | Reduce forward speed first, giving correction more time. Don’t immediately tighten the 5 cm arrival tolerance. |
| Near an endpoint: approach is too fast | Reduce distance_kp. This mainly affects the distance-controlled approach region. |
| Robot starts driving before sufficiently aligned | Reduce align_tolerance slightly. Keep it above heading_deadband. |
| Tiny heading noise causes repeated small turn commands | Increase heading_deadband slightly, accepting a little more residual heading error. |
| Rapid CARROT ↔ SEGMENT switching near the lateral threshold | Widen the gap between lateral_on and lateral_off; keep lateral_on > lateral_off. |