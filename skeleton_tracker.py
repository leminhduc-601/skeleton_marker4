#!/usr/bin/env python
from data_skeleton import initialize_camera, get_skeleton_coordinates,create_camera_box_marker, add_skeleton_collision_objects
import rospy
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import mediapipe as mp
import cv2
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_from_euler
            



translation = (1.1, 0, 1.5)
rotation_rpy = (-0.82, 0, 1.5708)

def publish_static_transform(translation, rotation_rpy):
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped = TransformStamped()

    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = "base_link"
    static_transformStamped.child_frame_id = "kinect_link"

    # Gán vị trí
    static_transformStamped.transform.translation.x = translation[0]
    static_transformStamped.transform.translation.y = translation[1]
    static_transformStamped.transform.translation.z = translation[2]

    # Gán hướng xoay (quaternion)
    quat = quaternion_from_euler(*rotation_rpy)
    static_transformStamped.transform.rotation.x = quat[0]
    static_transformStamped.transform.rotation.y = quat[1]
    static_transformStamped.transform.rotation.z = quat[2]
    static_transformStamped.transform.rotation.w = quat[3]

    broadcaster.sendTransform(static_transformStamped)



import time
def main():
    rospy.init_node('skeleton_publisher')
    publish_static_transform(translation, rotation_rpy)
    pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10, latch = True)
    image_pub = rospy.Publisher('/kinect_rgb/image_raw', Image, queue_size=10)
    bridge = CvBridge()
    rate = rospy.Rate(30)   # toc do xu ly cua camera
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=0, enable_segmentation=False)
    mp_drawing = mp.solutions.drawing_utils
    fn, device, listener, registration = initialize_camera()
    marker_array = MarkerArray()
    camera_box = create_camera_box_marker()
    marker_array.markers.append(camera_box)
    pub.publish(marker_array)

    try:
        while not rospy.is_shutdown():
            # 1. Lấy tọa độ skeleton
            skeleton_coordinates = get_skeleton_coordinates(listener, pose, mp_drawing, image_pub, bridge, registration)
            print(skeleton_coordinates)
            #2. Hiển thị khung xương như là một vật cản trong môi trường rviz
            add_skeleton_collision_objects(skeleton_coordinates)
            rate.sleep()


    except KeyboardInterrupt:
        print("Shutting down due to Ctrl+C...")
    finally:
        device.stop()
        device.close()
        cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()




























# def transform_skeleton_to_base(skeleton_coordinates, tf_listener):
#     """
#     Biến đổi tọa độ khung xương từ hệ 'kinect_link' sang 'base_link'.

#     Args:
#         skeleton_coordinates: dict {joint_id: (x, y, z)} trong hệ kinect_link
#         tf_listener: tf.TransformListener instance đã được khởi tạo

#     Returns:
#         dict {joint_id: (x, y, z)} trong hệ base_link
#     """
#     transformed_coordinates = {}

#     for joint_id, (x, y, z) in skeleton_coordinates.items():
#         point_in_kinect = PointStamped()
#         point_in_kinect.header.frame_id = "kinect_link"
#         point_in_kinect.header.stamp = rospy.Time(0)
#         point_in_kinect.point.x = x
#         point_in_kinect.point.y = y
#         point_in_kinect.point.z = z

#         try:
#             point_in_base = tf_listener.transformPoint("base_link", point_in_kinect)
#             transformed_coordinates[joint_id] = (
#                 point_in_base.point.x,
#                 point_in_base.point.y,
#                 point_in_base.point.z
#             )
#         except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
#             rospy.logwarn(f"Không thể transform joint {joint_id} từ kinect_link sang base_link")
#             transformed_coordinates[joint_id] = None  # hoặc có thể bỏ qua key này tùy mục đích

#     return transformed_coordinates

            # # 3. Cập nhật grid từ các khớp:
            # grid[:] = 0  # reset toàn bộ grid
            # transformed_coordinates = transform_skeleton_to_base(skeleton_coordinates, tf_listener)
            # mark_obstacle(grid, transformed_coordinates)
            # publish_voxels(marker_pub, grid, show_free = True)