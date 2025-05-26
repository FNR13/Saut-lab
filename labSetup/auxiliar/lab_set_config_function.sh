# Receive robot number (between 17 and 23)
export ROBOT_NUMBER=$1

export LOCAL_IP=$(hostname -I | awk '{print $1}')  # Extract the first IP address
  
# Set the ROS_MASTER_URI to the chosen robot's IP address
export ROS_MASTER_URI=http://192.168.28.$ROBOT_NUMBER:11311
export ROS_HOSTNAME=$LOCAL_IP
export ROS_IP=$LOCAL_IP

# Display confirmation
echo "ROS configurations set for robot $ROBOT_NUMBER."
echo "ROS_MASTER_URI=$ROS_MASTER_URI"
echo "ROS_HOSTNAME=$ROS_HOSTNAME"
echo "ROS_IP=$ROS_IP"