#!/usr/bin/env python3
import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from moveit_commander import PlanningSceneInterface, RobotCommander, MoveGroupCommander
from geometry_msgs.msg import Point, Quaternion, Pose
from Dstar_Lite import DStarLite, GRID_X, GRID_Y, GRID_Z, VOXEL_SIZE, X_RANGE, Y_RANGE, Z_RANGE
from tf.transformations import quaternion_from_euler


connection_pairs = [(14,16),(13,15)]


grid = np.zeros((GRID_X, GRID_Y, GRID_Z), dtype=int)

def world_to_voxel(x, y, z):
    i = int((x - X_RANGE[0]) / VOXEL_SIZE)
    j = int((y - Y_RANGE[0]) / VOXEL_SIZE)
    k = int((z - Z_RANGE[0]) / VOXEL_SIZE)
    return (i, j, k)

def voxel_to_world(i, j, k):
    x = i * VOXEL_SIZE + X_RANGE[0] + VOXEL_SIZE / 2
    y = j * VOXEL_SIZE + Y_RANGE[0] + VOXEL_SIZE / 2
    z = k * VOXEL_SIZE + Z_RANGE[0] + VOXEL_SIZE / 2
    return x, y, z

def create_voxel_marker(i, j, k, marker_id, start, goal, visited_path, barriers=None):
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "voxels"
    marker.id = marker_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = voxel_to_world(i, j, k)
    marker.pose.orientation.w = 1.0
    marker.scale.x = VOXEL_SIZE
    marker.scale.y = VOXEL_SIZE
    marker.scale.z = VOXEL_SIZE

    if barriers and (i, j, k) in barriers:
        marker.color = ColorRGBA(0.0, 0.0, 0.0, 1.0)  # đen đặc
    elif (i, j, k) == start:
        marker.color = ColorRGBA(1.0, 0.5, 0.0, 1.0)  # cam
    elif (i, j, k) == goal:
        marker.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)  # xanh lam
    elif visited_path and (i, j, k) in visited_path:
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.3)  # đỏ đậm
    else:
        marker.color = ColorRGBA(0.8, 0.8, 0.8, 0.1)  # xám nhạt
    return marker


def publish_voxels(marker_pub, start, goal, visited_path=None, barriers=None):
    marker_id = 0
    all_points = set()
    all_points.add(start)
    all_points.add(goal)
    if visited_path:
        all_points.update(visited_path)
    if barriers:
        all_points.update(barriers)

    for (i, j, k) in all_points:
        marker = create_voxel_marker(i, j, k, marker_id, start, goal, visited_path, barriers)
        marker_pub.publish(marker)
        marker_id += 1
        rospy.sleep(0.005)

    



def move_to_start(move_group, start_voxel):
    """
    Di chuyển robot đến vị trí voxel start với orientation ổn định nhất trước khi bắt đầu thuật toán.
    """
    # 1. Chuyển voxel sang tọa độ thế giới
    x, y, z = voxel_to_world(*start_voxel)

    # 2. Đặt orientation ổn định (EEF hướng xuống, phù hợp UR3)
    qx, qy, qz, qw = quaternion_from_euler(3.14, 0, -1.57)  # roll, pitch, yaw

    # 3. Tạo pose đích
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw

    # 4. Gán mục tiêu và lập kế hoạch
    move_group.set_pose_target(pose)
    plan = move_group.plan()

    # 5. Thực thi
    success = move_group.execute(plan[1], wait=True)

    if success:
        rospy.loginfo("✓ Di chuyển đến vị trí START thành công.")
        rospy.sleep(5)
    else:
        rospy.logwarn("✘ Di chuyển đến vị trí START thất bại.")

    move_group.clear_pose_targets()
    rospy.sleep(0.5)

def move_between_voxels(move_group, path):
    """
    Di chuyển robot UR3 theo danh sách các voxel bằng quỹ đạo Cartesian.
    Luôn thực hiện execute plan bất kể fraction.
    Trả về:
        - voxel_reached: ô cuối cùng robot đã thực sự đi tới được
        - voxel_blocked: ô robot định đi tiếp nhưng bị chặn (None nếu không bị chặn)
        - k_m: số voxel mà robot đã đi qua (≥ 1 nếu đi được ít nhất 1 voxel)
    """
    # 1. Lấy orientation hiện tại
    orientation = move_group.get_current_pose().pose.orientation

    # 2. Chuyển path voxel sang waypoint Pose (tọa độ thế giới)
    waypoints = []
    for voxel in path:
        x, y, z = voxel_to_world(*voxel)
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation = orientation
        waypoints.append(pose)

    if not waypoints:
        rospy.logwarn("✘ Không có waypoint nào để thực hiện.")
        return None, None, 0
    import time

    start = time.time()
    # 3. Tính toán quỹ đạo
    plan, fraction = move_group.compute_cartesian_path(
        waypoints,
        eef_step=0.05,
        avoid_collisions=True
    )
    end = time.time()
    print("Thời gian chạy:", (end - start) * 1000, "ms")

    # 4. Thực thi plan bất kể fraction
    move_group.execute(plan, wait=True)
    move_group.stop()
    move_group.clear_pose_targets()

    rospy.loginfo(f"✓ Đã thực hiện plan với tỉ lệ thành công: {fraction:.2f}")

    # 5. Tính số voxel đã đi được (k_m)
    num_waypoints = len(waypoints)
    k_m = int(fraction * num_waypoints)

    # 6. Xác định voxel_reached và voxel_blocked
    if k_m == 0:
        voxel_reached = path[0]
        voxel_blocked = path[0]
    else:
        voxel_reached = path[k_m - 1]
        voxel_blocked = None
        if k_m < len(path):
            voxel_blocked = path[k_m]

    if voxel_blocked:
        rospy.logwarn("Phát hiện va chạm với con người")
    return voxel_reached, voxel_blocked, k_m


def clear_markers(marker_pub, show_true=False):
    # 1. Xóa tất cả marker cũ
    delete_marker = Marker()
    delete_marker.action = Marker.DELETEALL
    delete_marker.header.frame_id = "base_link"
    delete_marker.header.stamp = rospy.Time.now()
    delete_marker.ns = "voxels"
    marker_pub.publish(delete_marker)
    rospy.sleep(0.05)

    # 2. Nếu bật hiển thị voxel xám toàn môi trường
    if show_true:
        marker_id = 500
        for i in range(GRID_X):
            for j in range(GRID_Y):
                for k in range(GRID_Z):
                    marker = Marker()
                    marker.header.frame_id = "base_link"
                    marker.header.stamp = rospy.Time.now()
                    marker.ns = "voxels"
                    marker.id = marker_id
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD
                    marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = voxel_to_world(i, j, k)
                    marker.pose.orientation.w = 1.0
                    marker.scale.x = VOXEL_SIZE
                    marker.scale.y = VOXEL_SIZE
                    marker.scale.z = VOXEL_SIZE
                    marker.color = ColorRGBA(0.8, 0.8, 0.8, 0.1)  # xám nhạt
                    marker_pub.publish(marker)
                    marker_id += 1
                    rospy.sleep(0.001)

from moveit_commander import PlanningSceneInterface
from geometry_msgs.msg import PoseStamped

def add_real_obstacle_between(pose1, pose2, size=0.15):
    """
    Tạo vật cản thật sự (collision object) giữa hai điểm pose1 và pose2.
    """
    planning_scene = PlanningSceneInterface(synchronous=True)
    rospy.sleep(2.0)  # Đảm bảo PlanningScene đã khởi động

    # Tạo PoseStamped với frame_id là base_link
    center_pose_stamped = PoseStamped()
    center_pose_stamped.header.frame_id = "base_link"
    center_pose_stamped.pose.position.x = (pose1.position.x + pose2.position.x) / 2
    center_pose_stamped.pose.position.y = (pose1.position.y + pose2.position.y) / 2
    center_pose_stamped.pose.position.z = (pose1.position.z + pose2.position.z) / 2
    center_pose_stamped.pose.orientation.w = 1.0

    planning_scene.add_box(
        name="middle_obstacle",
        pose=center_pose_stamped,
        size=(size, size, size)
    )
    rospy.loginfo("✓ Đã thêm vật cản thật sự giữa hai điểm.")

from moveit_commander import RobotCommander, PlanningSceneInterface
from moveit_msgs.srv import GetPlanningScene, ApplyPlanningScene
from moveit_msgs.msg import AllowedCollisionMatrix, AllowedCollisionEntry, PlanningScene
import rospy

def allow_collision_with_all_objects():
    robot = RobotCommander()
    scene = PlanningSceneInterface()
    rospy.sleep(1.0)  # Chờ để scene ổn định

    rospy.wait_for_service('/get_planning_scene')
    get_planning_scene = rospy.ServiceProxy('/get_planning_scene', GetPlanningScene)
    scene_resp = get_planning_scene()
    acm = scene_resp.scene.allowed_collision_matrix

    # Lấy danh sách tên object và link
    object_names = scene.get_known_object_names()
    link_names = robot.get_link_names()

    existing_names = set(acm.entry_names)

    # Bổ sung vào ACM nếu chưa có
    for name in object_names + link_names:
        if name not in existing_names:
            acm.entry_names.append(name)
            for entry in acm.entry_values:
                entry.enabled.append(True)
            new_entry = AllowedCollisionEntry(enabled=[True]*len(acm.entry_names))
            acm.entry_values.append(new_entry)
            existing_names.add(name)

    # Tạo bảng tra index
    name_index = {name: i for i, name in enumerate(acm.entry_names)}

    # Bật va chạm cho từng cặp link ↔ object
    for link in link_names:
        for obj in object_names:
            i = name_index[link]
            j = name_index[obj]
            acm.entry_values[i].enabled[j] = True
            acm.entry_values[j].enabled[i] = True

    # Gửi diff scene
    planning_scene_diff = PlanningScene()
    planning_scene_diff.is_diff = True
    planning_scene_diff.allowed_collision_matrix = acm

    rospy.wait_for_service('/apply_planning_scene')
    apply_planning_scene = rospy.ServiceProxy('/apply_planning_scene', ApplyPlanningScene)
    apply_planning_scene(planning_scene_diff)

    rospy.loginfo("✓ Đã bật cho phép va chạm giữa mọi link và mọi object trong PlanningScene.")

def main():
    visited_path = []

    rospy.init_node('robot_controller')
    pose_A = Pose(
        position=Point(-0.3, -0.25, 0.2),
        orientation=Quaternion(0.0, 1.0, 0.0, 0.0)
    )

    pose_B = Pose(
        position=Point(-0.3, 0.25, 0.2),
        orientation=Quaternion(0.0, 1.0, 0.0, 0.0)
    )
    # Chuyển từ tọa độ thế giới sang tọa độ voxel
    start_voxel = world_to_voxel(pose_A.position.x, pose_A.position.y, pose_A.position.z)
    goal_voxel = world_to_voxel(pose_B.position.x, pose_B.position.y, pose_B.position.z)
    robot = RobotCommander()
    move_group = MoveGroupCommander("manipulator")
    marker_pub = rospy.Publisher('voxel_markers', Marker, queue_size=1)

    rospy.sleep(1.0)

    # add_real_obstacle_between(pose_A, pose_B)
    rospy.sleep(1.0)
    # allow_collision_with_all_objects()


    
    

    s_last  = start_voxel
    k_m = 0
    planner = DStarLite(start_voxel, goal_voxel, k_m)
    planner.computeShortestPath(start_voxel, k_m)

    try:
        while not rospy.is_shutdown():
            move_to_start(move_group, start_voxel)
            clear_markers(marker_pub, show_true = False)
            while True:
                path = planner.reconstruct_path(s_last)
                publish_voxels(marker_pub, start_voxel, goal_voxel, path, planner.barriers)
                s_last, start, k_m = move_between_voxels(move_group, path)
                if s_last == goal_voxel:
                    rospy.loginfo("Đã di chuyển xong tới đích")
                    return
                else:
                    planner.barriers.add(start)
                    planner.updateVertex(start, s_last, k_m)
                    planner.computeShortestPath(s_last, k_m)
    except KeyboardInterrupt:
        print("Shutting down due to Ctrl+C...")
    


if __name__ == "__main__":
    main()

