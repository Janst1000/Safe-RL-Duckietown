cmake_minimum_required(VERSION 2.8.3)
project(rl_agent)

find_package(catkin REQUIRED COMPONENTS
  rospy
  duckietown_msgs # Every duckietown packages should use this.
  cv_bridge
  tf
  std_msgs
  message_generation
)

# include launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# add message files
add_message_files(
  FILES
  SafetyMsg.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
  std_msgs
)

# add python scripts
catkin_install_python(PROGRAMS
  src/collect_data.py
  src/deep_rl_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
