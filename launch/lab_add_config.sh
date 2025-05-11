#!/bin/bash

ROBOT_NUMBER=$1

if [[ -z "$ROBOT_NUMBER" || "$ROBOT_NUMBER" -lt 17 || "$ROBOT_NUMBER" -gt 23 ]]; then

  sed -i '\|source /home/ricardo/saut/launch/auxiliar/lab_set_config_function.sh|d' ~/.bashrc
  echo "Removed any previous ROS config sourcing lines from .bashrc"

  echo "If you want to set the config, Enter the robot number: ./script.sh <robot_number (17-23)>"

  exit 1
fi

# Script path without arguments
CONFIG_BASE="source /home/ricardo/saut/launch/auxiliar/lab_set_config_function.sh"

# Removing previous first
sed -i '\|source /home/ricardo/saut/launch/auxiliar/lab_set_config_function.sh|d' ~/.bashrc
echo "Removed any previous ROS config sourcing lines from .bashrc"
echo " "

# Setting to bashrc
echo "$CONFIG_BASE $ROBOT_NUMBER" >> ~/.bashrc

# Confirming on terminal
echo "Added ROS config sourcing to .bashrc:"
echo $CONFIG_BASE $ROBOT_NUMBER
