import cv2
import numpy as np
from pylibfreenect2 import Freenect2, Freenect2Device, FrameType, SyncMultiFrameListener, Frame, Registration 
import mediapipe as mp
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from tf.transformations import quaternion_about_axis
2456478576987869689896896896
# Các cặp khớp cần nối
connection_pairs = [(11,13),(13,15),(11,23),(11,12),(12,24),(23,24),(12,14),(14,16),(24,26),(26,28),(23,25),(25,27)]
def create_camera_box_marker():
    marker = Marker()
    marker.header.frame_id = "kinect_link"  # chính là khung tọa độ của camera
    marker.header.stamp = rospy.Time.now()
    marker.ns = "camera"
    marker.id = 999  # ID bất kỳ không trùng các khớp
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    # Vị trí khối box — (0, 0, 0) là gốc khung kinect_link
    marker.pose.position.x = 0.0
    marker.pose.position.y = 0.0
    marker.pose.position.z = 0.0    

    # Hướng không xoay
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    # Kích thước hình hộp — ví dụ 10cm x 5cm x 5cm
    marker.scale.x = 0.1
    marker.scale.y = 0.05
    marker.scale.z = 0.05                                                                                                                                                                                   

    # Màu sắc — xanh dương nhạt, có alpha
    marker.color.r = 0.0
    marker.color.g = 0.5
    marker.color.b = 1.0
    marker.color.a = 0.8

    marker.lifetime = rospy.Duration(0)  # tồn tại mãi

    return marker

# Danh sách các object đã tạo
known_objects = set()
collision_pub = rospy.Publisher('/collision_object', CollisionObject, queue_size=10)


def add_skeleton_collision_objects(skeleton_coordinates):
    current_ids = set()  # <<< SỬA TẠI ĐÂY: tạo danh sách ID hiện có lần này

    # --- Hiển thị các khớp ---
    for idx, (x, y, z) in skeleton_coordinates.items():
        obj_id = f"joint_{idx}"
        current_ids.add(obj_id)  # <<< SỬA TẠI ĐÂY

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [0.05]  # bán kính 5cm

        pose = PoseStamped()
        pose.header.frame_id = "kinect_link"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0

        obj = CollisionObject()
        obj.id = obj_id
        obj.header.frame_id = "kinect_link"
        obj.primitives.append(primitive)
        obj.primitive_poses.append(pose.pose)

        if obj_id not in known_objects:
            obj.operation = CollisionObject.ADD
            known_objects.add(obj_id)
        else:
            obj.operation = CollisionObject.MOVE

        collision_pub.publish(obj)

    # --- Hiển thị các xương ---
    for idx1, idx2 in connection_pairs:
        if idx1 in skeleton_coordinates and idx2 in skeleton_coordinates:
            p1 = np.array(skeleton_coordinates[idx1])
            p2 = np.array(skeleton_coordinates[idx2])
            center = (p1 + p2) / 2.0
            direction = p2 - p1
            height = np.linalg.norm(direction)
            if height == 0:
                continue

            axis = np.cross([0, 0, 1], direction)
            angle = np.arccos(np.dot([0, 0, 1], direction / height))
            if np.linalg.norm(axis) < 1e-6:
                quat = [0, 0, 0, 1] if direction[2] > 0 else [1, 0, 0, 0]
            else:
                quat = quaternion_about_axis(angle, axis / np.linalg.norm(axis))

            obj_id = f"bone_{idx1}_{idx2}"
            current_ids.add(obj_id)  # <<< SỬA TẠI ĐÂY

            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.CYLINDER
            primitive.dimensions = [height, 0.04]  # height, radius

            pose = PoseStamped()
            pose.header.frame_id = "kinect_link"
            pose.pose.position.x = center[0]
            pose.pose.position.y = center[1]
            pose.pose.position.z = center[2]
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]

            obj = CollisionObject()
            obj.id = obj_id
            obj.header.frame_id = "kinect_link"
            obj.primitives.append(primitive)
            obj.primitive_poses.append(pose.pose)

            if obj_id not in known_objects:
                obj.operation = CollisionObject.ADD
                known_objects.add(obj_id)
            else:
                obj.operation = CollisionObject.MOVE

            collision_pub.publish(obj)

    # --- Xóa các object không còn trong frame hiện tại ---
    to_remove = known_objects - current_ids  # <<< SỬA TẠI ĐÂY

    for obj_id in to_remove:
        obj = CollisionObject()
        obj.id = obj_id
        obj.header.frame_id = "kinect_link"
        obj.operation = CollisionObject.REMOVE
        collision_pub.publish(obj)

    known_objects.intersection_update(current_ids)  # <<< SỬA TẠI ĐÂY: cập nhật lại danh sách

def get_skeleton_coordinates(listener,pose,mp_drawing,image_pub, bridge,registration):
    # Đợi và lấy frame mới
    frames = listener.waitForNewFrame()

    color_frame = frames[FrameType.Color]
    depth_frame = frames[FrameType.Depth]
    # Tạo khung chứa kết quả đăng ký
    undistorted = Frame(512, 424, 4)
    registered = Frame(512, 424, 4)
    # Đăng ký ảnh: căn chỉnh ảnh màu với ảnh depth
    registration.apply(color_frame, depth_frame, undistorted, registered)
    registered_image = registered.asarray(dtype=np.uint8)
    registered_image_bgr = cv2.cvtColor(registered_image, cv2.COLOR_RGB2BGR)

    # === THÊM: Nhận diện khung xương bằng Mediapipe Pose ===
    # Chuyển ảnh sang RGB để dùng với Mediapipe
    results = pose.process(registered_image_bgr)
    skeleton_coordinates = {}
    skeleton_coordinates2 = {}
    required_landmarks = [0, 11, 13, 15, 12, 14, 16, 23, 24, 25, 26, 27, 28]
    # Vẽ khung xương nếu tìm thấy
    if results.pose_landmarks:
        for idx in required_landmarks:
            landmark = results.pose_landmarks.landmark[idx]
            x_px = int(landmark.x * registered.width)
            y_px = int(landmark.y * registered.height)
            visibility = landmark.visibility
            if visibility > 0.90:
                point_3d = registration.getPointXYZ(undistorted, y_px, x_px)
                x, y, z = point_3d
                if not any(np.isnan([x, y, z])):
                    print(f"Landmark {idx}: (x={x}, y={y}, z={z:.4f}, visibility={visibility})")
                    skeleton_coordinates[idx] = (x, z, -y)
            mp_drawing.draw_landmarks(
                registered_image_bgr,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
            )

    # if 23 in skeleton_coordinates and 24 in skeleton_coordinates:
    #     hip_left = np.array(skeleton_coordinates[23])
    #     hip_right = np.array(skeleton_coordinates[24])
    #     hip_center = (hip_left + hip_right) / 2.0
    #     x1, y1, z1 = hip_center
    # else:
    #     hip_center = None  # Hoặc xử lý fallback sau
    # #print(f"hip_center: {hip_center}")
    # if results.pose_world_landmarks:
    #     for idx, landmark2 in enumerate(results.pose_world_landmarks.landmark):
    #         if idx in required_landmarks:
    #             x, y, z = landmark2.x, landmark2.y, landmark2.z
    #             visibility = landmark2.visibility
    #             if visibility > 0.6 and hip_center is not None:
    #                 skeleton_coordinates2[idx] = (-x - x1, z + y1, -y + z1) 
    #                 # print(f"Landmark {idx}: (x={x}, y={y}, z={z:.4f}, visibility={visibility})")

    # # Hiển thị
    ros_image = bridge.cv2_to_imgmsg(registered_image_bgr, encoding="bgr8")
    image_pub.publish(ros_image)
    
    listener.release(frames)
    return skeleton_coordinates, skeleton_coordinates2 


def initialize_camera():
    fn = Freenect2()
    num_devices = fn.enumerateDevices()
    if num_devices == 0:
        print("Không tìm thấy thiết bị Kinect v2")
        exit(1)
    serial = fn.getDeviceSerialNumber(0)
    device = fn.openDevice(serial)
    listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)
    device.setColorFrameListener(listener)
    device.setIrAndDepthFrameListener(listener)
    device.start()

    registration = Registration(
    device.getIrCameraParams(),
    device.getColorCameraParams()
)
    print("Streaming... Nhấn ESC để thoát")

    return fn, device, listener, registration
    








































# def create_marker(marker_id, point):
#     marker = Marker()
#     marker.header.frame_id = "kinect_link"
#     marker.header.stamp = rospy.Time.now()
#     marker.ns = "skeleton"
#     marker.id = marker_id
#     marker.type = Marker.SPHERE
#     marker.action = Marker.ADD
#     marker.pose.position.x = point[0]
#     marker.pose.position.y = point[1]
#     marker.pose.position.z = point[2]
#     marker.pose.orientation.w = 1.0
#     marker.scale.x = 0.05
#     marker.scale.y = 0.05
#     marker.scale.z = 0.05
#     marker.color.g = 1.0
#     marker.color.a = 1.0
#     marker.lifetime = rospy.Duration()
#     return marker



# def create_skeleton_lines(skeleton_coordinates):
#     marker = Marker()
#     marker.header.frame_id = "kinect_link"
#     marker.header.stamp = rospy.Time.now()
#     marker.ns = "skeleton_lines"
#     marker.id = 1000
#     marker.type = Marker.LINE_LIST
#     marker.action = Marker.ADD
#     marker.scale.x = 0.01
#     marker.color.r = 1.0
#     marker.color.g = 1.0
#     marker.color.b = 0.0
#     marker.color.a = 1.0
#     marker.lifetime = rospy.Duration()

#     for idx1, idx2 in connection_pairs:
#         # Kiểm tra cả hai điểm đều tồn tại trong skeleton_coordinates
#         if {idx1, idx2} <= skeleton_coordinates.keys():
#             p1 = skeleton_coordinates[idx1]
#             p2 = skeleton_coordinates[idx2]
#             marker.points.append(Point(x=p1[0], y=p1[1], z=p1[2]))
#             marker.points.append(Point(x=p2[0], y=p2[1], z=p2[2]))
#         else:
#             continue  # Bỏ qua nếu 1 trong 2 điểm chưa được nhận diện

#     return marker
