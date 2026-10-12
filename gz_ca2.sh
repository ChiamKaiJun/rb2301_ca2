#!/bin/bash
source install/setup.bash
ros2 launch rb2301_gz ca2_gazebo.launch.py x:=-0.1 y:=0.1
pkill -9 ruby # kills any lingering Gazebo processes
