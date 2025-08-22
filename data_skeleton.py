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

# ====== SIMPLE ARM FILTERS: outlier + median + bone-length lock ======
from collections import defaultdict, deque
import numpy as np

# ---- Tham số chỉnh được ----
WINDOW = 5           # cửa sổ median (5 là hợp lý: mượt nhẹ, trễ ~2 frame)
MAX_JUMP = 0.5       # bước nhảy tối đa chấp nhận giữa 2 frame (đơn vị VỊ TRÍ của bạn: m hoặc mm)
CALIB_FRAMES = 50    # số frame để hiệu chuẩn độ dài xương tay
# ----------------------------

# Lịch sử dữ liệu từng khớp (để tính median theo trục)
_joint_hist = defaultdict(lambda: deque(maxlen=WINDOW))

# Bộ đếm frame & trạng thái hiệu chuẩn xương tay
_frame_count = 0
_calibrated = False

# Danh sách xương tay cần cố định
ARM_BONES = [
    (12, 14),  # vai trái - khuỷu trái
    (14, 16),  # khuỷu trái - cổ tay trái
    (11, 13),  # vai phải - khuỷu phải
    (13, 15),  # khuỷu phải - cổ tay phải
]

# Lưu mẫu độ dài trong giai đoạn hiệu chuẩn & độ dài chuẩn sau khi chốt
_bone_samples = defaultdict(lambda: deque(maxlen=CALIB_FRAMES))
_bone_length = {}  # {(a,b): L}

def _dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def _median3(hist):
    xs = sorted(h[0] for h in hist)
    ys = sorted(h[1] for h in hist)
    zs = sorted(h[2] for h in hist)
    m = len(hist) // 2
    return (xs[m], ys[m], zs[m])

def _filter_one_joint(jid, p):
    """
    Bước 1: loại outlier mạnh so với frame trước (MAX_JUMP)
    Bước 2: thêm vào lịch sử & trả về median theo trục
    """
    hist = _joint_hist[jid]
    if hist:
        prev = hist[-1]
        if _dist(p, prev) > MAX_JUMP:
            # outlier → bỏ frame này, dùng lại giá trị trước
            return prev

    # ok → nhận frame này, thêm vào lịch sử
    hist.append(p)
    return _median3(hist)

def _calibrate_arm_lengths(skel):
    """
    Thu thập độ dài xương tay trong CALIB_FRAMES frame đầu.
    Dùng median để chốt độ dài chuẩn.
    Chỉ bắt đầu đếm frame khi có đủ khớp tay trong skeleton.
    """
    global _frame_count, _calibrated

    if _calibrated:
        return

    # 1. Kiểm tra có đủ khớp của xương tay chưa
    valid = all(a in skel and b in skel for a, b in ARM_BONES)
    if not valid:
        return  # chưa đủ → bỏ qua frame này

    # 2. Nếu đủ → thu mẫu độ dài xương tay (chỉ nhận mẫu hợp lý)
    for a, b in ARM_BONES:
        d = _dist(skel[a], skel[b])
        if 0.1 < d < 1.0:  # khoảng hợp lý với người
            _bone_samples[(a, b)].append(d)

    _frame_count += 1

    # 3. Khi đủ số frame → chốt median
    if _frame_count >= CALIB_FRAMES:
        for k, samples in _bone_samples.items():
            if samples:
                srt = sorted(samples)
                _bone_length[k] = srt[len(srt)//2]
        _calibrated = True
        print("[INFO] ✅ Bone calibration locked — used", _frame_count, "frames")

def _fix_bone_lengths(skel):
    """
    Chuẩn hóa lại độ dài xương tay về đúng chiều dài đã hiệu chuẩn.
    Chỉnh bằng cách giữ nguyên trung điểm cặp khớp và kéo 2 đầu về đúng L/2.
    """
    if not _bone_length:
        return skel

    fixed = dict(skel)
    for (a, b), L in _bone_length.items():
        if a in fixed and b in fixed:
            pa = np.array(fixed[a], dtype=float)
            pb = np.array(fixed[b], dtype=float)
            v = pb - pa
            n = np.linalg.norm(v)
            if n < 1e-6 or L <= 0:
                continue
            mid = (pa + pb) / 2.0
            vu = v / n
            pa_new = mid - vu * (L / 2.0)
            pb_new = mid + vu * (L / 2.0)
            fixed[a] = (float(pa_new[0]), float(pa_new[1]), float(pa_new[2]))
            fixed[b] = (float(pb_new[0]), float(pb_new[1]), float(pb_new[2]))
    return fixed

def arm_filter_pipeline(skeleton_coordinates):
    """
    Pipeline gọn để gọi trước khi return:
      Raw → (outlier check) → (median) → (lock bone length sau calib)
    """
    global _calibrated

    if not isinstance(skeleton_coordinates, dict):
        return skeleton_coordinates

    # 1) Lọc theo joint: outlier → bỏ, median smoothing
    filtered = {}
    for jid, p in skeleton_coordinates.items():
        filtered[jid] = _filter_one_joint(jid, p)

    # 2) Hiệu chuẩn độ dài xương tay (trong 50 frame đầu)
    if not _calibrated:
        _calibrate_arm_lengths(filtered)

    # 3) Nếu đã chốt chiều dài xương → sửa lại cho khớp tay
    if _calibrated:
        filtered = _fix_bone_lengths(filtered)

    return filtered
# ====== END SIMPLE ARM FILTERS ======



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

def get_skeleton_coordinates(listener, pose, mp_drawing, image_pub, bridge, registration):
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

    # === Nhận diện khung xương bằng Mediapipe Pose ===
    results = pose.process(registered_image_bgr)
    skeleton_coordinates = {}
    required_landmarks = [0, 11, 13, 15, 12, 14, 16, 23, 24, 25, 26, 27, 28]

    if results.pose_landmarks:
        filtered_landmarks = landmark_pb2.NormalizedLandmarkList()
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            if lm.visibility > 0.9 and idx in required_landmarks:
                x_px = int(lm.x * registered.width)
                y_px = int(lm.y * registered.height)
                point_3d = registration.getPointXYZ(undistorted, y_px, x_px)
                x, y, z = point_3d
                if not any(np.isnan([x, y, z])):
                    skeleton_coordinates[idx] = (x, z, -y)
                filtered_landmarks.landmark.append(lm)
            else:
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

    # === CHỈNH SỬA: ép y của 23 = y của 11, và y của 24 = y của 12 ===
    if 23 in skeleton_coordinates and 11 in skeleton_coordinates:
        x23, y23, z23 = skeleton_coordinates[23]
        _, y11, _ = skeleton_coordinates[11]
        skeleton_coordinates[23] = (x23, y11+ 0.3, z23)

    if 24 in skeleton_coordinates and 12 in skeleton_coordinates:
        x24, y24, z24 = skeleton_coordinates[24]
        _, y12, _ = skeleton_coordinates[12]
        skeleton_coordinates[24] = (x24, y12 + 0.3, z24)

    # Xuất ảnh ROS
    ros_image = bridge.cv2_to_imgmsg(registered_image_bgr, encoding="bgr8")
    image_pub.publish(ros_image)
    
    listener.release(frames)
    skeleton_coordinates = arm_filter_pipeline(skeleton_coordinates)
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
    








