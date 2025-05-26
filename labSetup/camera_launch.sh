#!/bin/bash

# Check if roscore is running
if rosnode list > /dev/null; then
  echo "roscore is running, proceeding with camera launch..."

  # Set the parameter
  rosparam set cv_camera/device_id 4

  # Run the camera node
  rosrun cv_camera cv_camera_node
else
  echo "roscore is not running. Please start roscore first."
fi
