cmake_minimum_required(VERSION 2.8.3)
project(autoware_tracker)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")

find_package(catkin REQUIRED COMPONENTS
    tf
    pcl_ros
    roscpp
    std_msgs
    sensor_msgs
    geometry_msgs
    visualization_msgs
    cv_bridge
    image_transport
    jsk_recognition_msgs
    jsk_rviz_plugins
    message_generation
)

# Messages
add_message_files(
    DIRECTORY msg
    FILES
        Centroids.msg
        CloudCluster.msg
        CloudClusterArray.msg
        DetectedObject.msg
        DetectedObjectArray.msg
)
generate_messages(
    DEPENDENCIES
      geometry_msgs
      jsk_recognition_msgs
      sensor_msgs
      std_msgs
)

catkin_package(
    #INCLUDE_DIRS include
    CATKIN_DEPENDS
    tf
    pcl_ros
    roscpp
    std_msgs
    sensor_msgs
    geometry_msgs
    visualization_msgs
    cv_bridge
    image_transport
    jsk_recognition_msgs
    jsk_rviz_plugins
    message_runtime
    message_generation
)


find_package(OpenMP)
find_package(OpenCV REQUIRED)
find_package(Eigen3 QUIET)

include_directories(
    #include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(visualize_detected_objects
    src/detected_objects_visualizer/visualize_detected_objects_main.cpp
    src/detected_objects_visualizer/visualize_detected_objects.cpp
    )
add_dependencies(visualize_detected_objects ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(visualize_detected_objects ${OpenCV_LIBRARIES} ${EIGEN3_LIBRARIES} ${catkin_LIBRARIES})

add_executable(lidar_euclidean_cluster_detect
    src/lidar_euclidean_cluster_detect/lidar_euclidean_cluster_detect.cpp
    src/lidar_euclidean_cluster_detect/cluster.cpp)
add_dependencies(lidar_euclidean_cluster_detect ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(lidar_euclidean_cluster_detect ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

add_executable(imm_ukf_pda
    src/lidar_imm_ukf_pda_track/imm_ukf_pda_main.cpp
    src/lidar_imm_ukf_pda_track/imm_ukf_pda.h
    src/lidar_imm_ukf_pda_track/imm_ukf_pda.cpp
    src/lidar_imm_ukf_pda_track/ukf.cpp
    )
add_dependencies(imm_ukf_pda ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(imm_ukf_pda ${catkin_LIBRARIES})
