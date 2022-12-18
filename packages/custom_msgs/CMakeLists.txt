cmake_minimum_required(VERSION 2.8.3)
project(custom_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  LaneDetection.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  #ChangePattern.srv
)


## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES duckietown_msgs
   CATKIN_DEPENDS
#  DEPENDS system_lib
)


include_directories(
  ${catkin_INCLUDE_DIRS}
)
