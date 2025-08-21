import cv2
import numpy as np
from pylibfreenect2 import Freenect2, Freenect2Device, FrameType, SyncMultiFrameListener, Frame, Registration 
import mediapipe as mp
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from tf.transformations import quaternion_about_axis
from mediapipe.framework.formats import landmark_pb2

class KalmanCV3D:
    """
    Kalman 3D với mô hình vận tốc hằng.
    Trạng thái: [x y z vx vy vz]^T
    - Gating (Mahalanobis) chống outlier mạnh
    - Lookahead giảm trễ cảm nhận
    """
    def __init__(self, process_var=1e-2, meas_var=5e-3,
                 gating_thresh=9.0, lookahead=0.033):
        self.x = None          # state 6x1
        self.P = None          # covariance 6x6
        self.q = process_var   # process noise power
        self.r = meas_var      # measurement noise
        self.last_t = None
        self.gating_thresh = gating_thresh
        self.lookahead = lookahead

    def _F_Q(self, dt: float):
        F = np.eye(6)
        F[0,3] = F[1,4] = F[2,5] = dt
        # Q ~ G q G^T (white accel model “nghèo” nhưng hiệu quả)
        G = np.array([[0.5*dt*dt],[0.5*dt*dt],[0.5*dt*dt],[dt],[dt],[dt]])
        Q = self.q * (G @ G.T)
        return F, Q

    def predict(self, dt: float):
        F, Q = self._F_Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def _out_with_lookahead(self):
        x, y, z, vx, vy, vz = self.x.flatten()
        return (float(x + vx*self.lookahead),
                float(y + vy*self.lookahead),
                float(z + vz*self.lookahead))

    def update(self, z, timestamp=None):
        now = time.time() if timestamp is None else float(timestamp)

        if self.last_t is None:
            # init
            self.last_t = now
            self.x = np.zeros((6,1))
            self.x[0,0], self.x[1,0], self.x[2,0] = z
            self.P = np.eye(6) * 1e-2
            return tuple(z)

        dt = max(1e-6, now - self.last_t)
        self.last_t = now

        # predict
        self.predict(dt)

        # measurement model
        H = np.zeros((3,6)); H[0,0]=H[1,1]=H[2,2]=1.0
        R = np.eye(3) * self.r
        z = np.array(z, dtype=float).reshape(3,1)

        # innovation + gating (Mahalanobis)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # phòng hờ trường hợp S suy biến
            Sinv = np.linalg.pinv(S)
        d2 = float(y.T @ Sinv @ y)  # ~ χ²(df=3)

        if d2 <= self.gating_thresh:
            # update bình thường
            K = self.P @ H.T @ Sinv
            self.x = self.x + K @ y
            I = np.eye(6)
            self.P = (I - K @ H) @ self.P
        # else: bỏ đo → giữ nguyên dự đoán

        return self._out_with_lookahead()

# ---- cấu hình bộ lọc & tiện ích dùng lại giữa các frame ----
FILTER_MODE = "kalman"   # đổi "off" nếu muốn tắt nhanh
KALMAN_PARAMS = dict(process_var=1e-2, meas_var=5e-3,
                     gating_thresh=9.0, lookahead=0.033)

_joint_kalman = {}   # lưu một bộ lọc cho mỗi joint-id

# (tùy chọn) kẹp bước nhảy để “chặn” spike còn sót
_MAX_STEP = 0.5      # mét / frame; chỉnh 0.3–0.7 tùy đơn vị của bạn
_last_pos = {}

def _clamp_step(jid, p):
    lp = _last_pos.get(jid)
    if lp is None:
        _last_pos[jid] = p
        return p
    v = np.array(p) - np.array(lp)
    n = float(np.linalg.norm(v))
    if n > _MAX_STEP and n > 1e-9:
        p = tuple((np.array(lp) + v * (_MAX_STEP/n)).tolist())
    _last_pos[jid] = p
    return p

def _get_kalman(jid):
    f = _joint_kalman.get(jid)
    if f is None:
        f = KalmanCV3D(**KALMAN_PARAMS)
        _joint_kalman[jid] = f
    return f
# ====== HẾT PHẦN KALMAN ======

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


collision_pub = rospy.Publisher('/collision_object', CollisionObject, queue_size=10)

def add_skeleton_collision_objects(skeleton_coordinates):
    # Tạo 1 CollisionObject duy nhất
    skeleton_obj = CollisionObject()
    skeleton_obj.id = "human_skeleton"
    skeleton_obj.header.frame_id = "kinect_link"
    skeleton_obj.operation = CollisionObject.ADD  # Luôn ADD để overwrite

    # --- Thêm khớp (hình cầu) ---
    for idx, (x, y, z) in skeleton_coordinates.items():
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [0.05]  # bán kính 5cm

        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = 1.0

        skeleton_obj.primitives.append(primitive)
        skeleton_obj.primitive_poses.append(pose)

    # --- Thêm xương (hình trụ) ---
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

            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.CYLINDER
            primitive.dimensions = [height, 0.04]  # height, radius

            pose = Pose()
            pose.position.x = center[0]
            pose.position.y = center[1]
            pose.position.z = center[2]
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]

            skeleton_obj.primitives.append(primitive)
            skeleton_obj.primitive_poses.append(pose)

    # --- Publish tất cả 1 lần ---
    collision_pub.publish(skeleton_obj)


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
        # Tạo bản sao đã lọc theo visibility
        filtered_landmarks = landmark_pb2.NormalizedLandmarkList()
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            if lm.visibility > 0.8  and idx in required_landmarks:
                x_px = int(lm.x * registered.width)
                y_px = int(lm.y * registered.height)
                point_3d = registration.getPointXYZ(undistorted, y_px, x_px)
                x, y, z = point_3d
                if not any(np.isnan([x, y, z])):
                    skeleton_coordinates[idx] = (x, z, -y)
                filtered_landmarks.landmark.append(lm)
            else:
                # Đẩy điểm ra ngoài khung hình để không vẽ
                fake = landmark_pb2.NormalizedLandmark(x=-1.0, y=-1.0, z=0.0, visibility=0.0)
                filtered_landmarks.landmark.append(fake)

        # Vẽ chỉ những điểm đủ điều kiện
        mp_drawing.draw_landmarks(
            registered_image_bgr,
            filtered_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
        )
    ros_image = bridge.cv2_to_imgmsg(registered_image_bgr, encoding="bgr8")
    image_pub.publish(ros_image)
    
    listener.release(frames)
        # === Áp dụng Kalman + Clamp Step cho từng joint ===
    if FILTER_MODE == "kalman" and isinstance(skeleton_coordinates, dict):
        filtered = {}
        now_ts = None   # có thể thay bằng rospy.get_time() nếu dùng ROS
        for jid, p in skeleton_coordinates.items():
            # Bảo hiểm: chặn bước nhảy phi lý trước khi đưa vào Kalman
            p = _clamp_step(jid, p)

            # Lấy Kalman filter cho khớp này
            kf = _get_kalman(jid)
            filtered[jid] = kf.update(p, timestamp=now_ts)

        skeleton_coordinates = filtered

    return skeleton_coordinates


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
