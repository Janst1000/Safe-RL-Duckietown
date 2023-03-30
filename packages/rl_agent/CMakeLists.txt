cmake_minimum_required(VERSION 2.8.3)
project(rl_agent)

find_package(catkin REQUIRED COMPONENTS
  rospy
  duckietown_msgs # Every duckietown packages should use this.
  cv_bridge
  tf
)

catkin_package()

# include launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# add python scripts
catkin_install_python(PROGRAMS
  src/rl_agent.py
  src/collect_data.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
