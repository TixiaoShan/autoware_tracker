# Readme

Barebone package for point cloud object tracking used in Autoware. The package is only tested in Ubuntu 16.04 and ROS Kinetic.

# Install JSK
```
sudo apt-get install ros-kinetic-jsk-recognition-msgs
sudo apt-get install ros-kinetic-jsk-rviz-plugins
```

# Compile
```
cd ~/catkin_ws/src
git clone https://github.com/TixiaoShan/autoware_tracker.git
cd ~/catkin_ws
catkin_make -j1
```
```-j1``` is only needed for message generation in the first install.

# Sample data

In case you don't have some bag files handy, you can download a sample bag using:
```
wget https://autoware-ai.s3.us-east-2.amazonaws.com/sample_moriyama_150324.tar.gz
```

# Demo

Run the autoware tracker:
```
roslaunch autoware_tracker run.launch
```

Play the sample ros bag:
```
rosbag play sample_moriyama_150324.bag
```