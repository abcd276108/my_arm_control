#!/usr/bin/env python3
# encoding: utf-8

import os
import cv2
import time
import yaml
import rclpy
import queue
import threading
import numpy as np

# 導入輔助運算與影像轉換模組
import sdk.common as common
from rclpy.node import Node
from app.utils import search_plane  # 深度相機平面尋找工具
from cv_bridge import CvBridge     # ROS 影像與 OpenCV 影像格式轉換

# 導入 ROS 2 標準訊息與服務
from std_msgs.msg import Bool
from dt_apriltags import Detector  # AprilTag 標籤偵測器
from sensor_msgs.msg import Image, CameraInfo 
from kinematics_msgs.srv import GetRobotPose  # 機械臂 Kinematics 正向/逆向運動學服務
from servo_controller_msgs.msg import ServosPosition
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from servo_controller.bus_servo_control import set_servo_position  # 機械臂馬達姿態控制
from tf2_ros import Buffer, TransformListener, TransformException  # ROS 2 座標轉換樹 (TF)


class StandaloneCalibrationNode(Node):
    """
    獨立運作的手眼標定與視覺區域對齊節點 (Standalone Eye-in-Hand Calibration Node)
    本腳本不需要 APP 服務發送 Trigger，啟動後會自動進行標定並將矩陣寫入 YAML 設定檔。
    """
    
    # 預設的手臂末端到相機的固定幾何矩陣 (Initial Hand-to-Camera TF Matrix)
    hand2cam_tf_matrix = [
            [0.0, 0.0, 1.0, -0.101],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.05],
            [0.0, 0.0, 0.0, 1.0]
            ]
    
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        
        # --- 狀態標誌與變數初始化 ---
        self.running = True        # 執行緒運作開關
        self.enter = True          # [修改] 預設直接進入標定狀態，無需等待 APP Trigger
        self.imgpts = None         # 繪製邊框用的 2D 圖像點
        self.imgpts1 = None
        self.plane = []
        self._init_parameters()
        
        # --- 標定板 (AprilTag) 參數設定 ---
        self.tag_size = 0.025      # AprilTag 邊長：2.5 cm (0.025 m)
        self.tag_id = [1, 2, 3]    # 允許標定的 Tag ID 列表
        self.tag_id_2 = 100
        
        # --- 環境變數與路徑讀取 ---
        self.camera_type = os.environ.get('CAMERA_TYPE', 'GEMINI')   # 相機類型 (GEMINI 或 USB_CAM)
        self.chassis_type = os.environ.get('CHASSIS_TYPE', 'Arm')     # 機構底盤類型
        self.config_file = 'transform.yaml'                          # 輸出手眼矩陣的檔案名稱
        
        if self.chassis_type == 'Slide_Rails':
            self.config_path = "/home/ubuntu/ros2_ws/src/stepper/config/"
        else:
            self.config_path = "/home/ubuntu/ros2_ws/src/app/config/"
            
        self.white_area_width = 0.167   # 白區/標定工作區寬度 (m)
        self.white_area_height = 0.13   # 白區/標定工作區高度 (m)
        
        # --- 影像佇列與轉換器 ---
        self.bridge = CvBridge()
        self.image_queue = queue.Queue(maxsize=2)
        self.depth_image_queue = queue.Queue(maxsize=2)

        # --- 建立 ROS 2 發布者 (Publishers) ---
        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1)        # 馬達控制
        self.result_image_pub = self.create_publisher(Image, '~/image_result', 10)               # 標定疊加影像
        self.finish_pub = self.create_publisher(Bool, '~/finish', 1)                             # 標定完成通知

        # --- 建立 AprilTag 標籤偵測物件 ---
        self.at_detector = Detector(searchpath=['apriltags'],
                       families='tag36h11',
                       nthreads=4,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)
        
        # --- 讀取深度相機與彩色相機之間的靜態 TF 轉換 ---
        tf_buffer = Buffer()
        self.tf_listener = TransformListener(tf_buffer, self)
        tf_future = tf_buffer.wait_for_transform_async(
            target_frame='depth_cam_link',
            source_frame='depth_cam_color_frame',
            time=rclpy.time.Time()
        )

        # 等待 TF 取得，計算相機間的外參轉換
        rclpy.spin_until_future_complete(self, tf_future)
        try:
            transform = tf_buffer.lookup_transform(
                'depth_cam_color_frame', 'depth_cam_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=5.0))
            self.static_transform = transform
        except TransformException as e:
            self.get_logger().error(f'無法取得 static transform: {e}')

        # 將 TF 轉為 4x4 齊次轉換矩陣 (Homogeneous Transformation Matrix)
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        transform_matrix = common.xyz_quat_to_mat([translation.x, translation.y, translation.z],
                                                  [rotation.w, rotation.x, rotation.y, rotation.z])
        
        # 更新 End-Effector 到 Camera 的整體矩陣
        self.hand2cam_tf_matrix = np.matmul(transform_matrix, self.hand2cam_tf_matrix)
        self.get_logger().info(f'校正後的 hand2cam_tf_matrix:\n{self.hand2cam_tf_matrix}')

        # 讀取現有的 YAML 檔 (若存在)
        if os.path.exists(self.config_path + self.config_file):
            with open(self.config_path + self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                self.extristric = np.array(config['extristric'])
                self.white_area_pose_cam = np.array(config['white_area_pose_cam'])
                self.white_area_pose_world = np.array(config['white_area_pose_world'])

        # --- [修改] 自動開啟 ROS 2 影像 Topic 訂閱與手臂姿勢初始化 ---
        self._subscribe_camera_topics()
        
        # 控制手臂移動到預設的標定觀測姿勢 (Move arm to calibration pose)
        set_servo_position(self.joints_pub, 1.5, ((1, 500), (2, 520), (3, 210), (4, 50), (5, 500), (10, 500)))

        # --- 啟動背景執行緒：影像處理與標定主流程 ---
        threading.Thread(target=self.image_processing, daemon=True).start()  # 圖像識別線程
        
        # [修改] 直接開啟一個 Thread 自動執行標定算碼，不需等待 Service 呼叫
        self.thread = threading.Thread(target=self.calibration_proc, daemon=True)
        self.thread.start()

    def _init_parameters(self):
        """初始化內部數據變數"""
        self.thread = None
        self.err_msg = None
        self.calibration_step = 0
        self.tags = []
        self.pose = []
        self.tag_count = 0
        self.K = None      # 相機內參矩陣 (Camera Matrix)
        self.D = None      # 相機畸變參數 (Distortion Coefficients)

        self.image_sub = None
        self.camera_info_sub = None
        self.depth_image_sub = None
        self.depth_cam_info = None
        self.depth_camera_info_sub = None

    def _subscribe_camera_topics(self):
        """[新增] 自動訂閱相機相關的 ROS Topics"""
        if self.camera_type == "GEMINI":
            self.image_sub = self.create_subscription(Image, '/depth_cam/rgb/image_raw', self.image_callback, 1)
            self.camera_info_sub = self.create_subscription(CameraInfo, '/depth_cam/rgb/camera_info', self.camera_info_callback, 1)
            self.depth_image_sub = self.create_subscription(Image, '/depth_cam/depth/image_raw', self.depth_image_callback, 1)
            self.depth_camera_info_sub = self.create_subscription(CameraInfo, '/depth_cam/depth/camera_info', self.depth_camera_info_callback, 1)
        elif self.camera_type == "USB_CAM":
            self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 1)
            self.camera_info_sub = self.create_subscription(CameraInfo, '/depth_cam/rgb/camera_info', self.camera_info_callback, 1)

    def calibration_proc(self):
        """
        手眼標定與座標系擬合核心算碼 (Hand-Eye Calibration Logic)
        """
        self.get_logger().info("=== 開始自動手眼標定流程 ===")

        # 1. 呼叫 Kinematics 服務，向手臂控制器查詢「當前末端執行器在世界座標系下的姿態 (${}^B T_E$)」
        timer_cb_group = ReentrantCallbackGroup()
        self.get_current_pose_client = self.create_client(GetRobotPose, '/kinematics/get_current_pose', callback_group=timer_cb_group)
        self.get_current_pose_client.wait_for_service()
        endpoint_res = self.send_request(self.get_current_pose_client, GetRobotPose.Request())
        
        pose_t = endpoint_res.pose.position
        pose_r = endpoint_res.pose.orientation
        # 將末端的平移與四元數轉為 4x4 齊次變換矩陣 (${}^B T_E$)
        endpoint = common.xyz_quat_to_mat([pose_t.x, pose_t.y, pose_t.z], [pose_r.w, pose_r.x, pose_r.y, pose_r.z])
            
        # 2. 觸發 AprilTag 採樣 (進入 step 1，等待影像線程收集 10 幀穩定標籤數據)
        t = time.time()
        self.tags = []
        self.calibration_step = 1  # 告訴 image_processing 開始收集 tags
        
        # 最多等待 15 秒採樣標籤
        while self.calibration_step == 1 and time.time() - t < 15:
            time.sleep(0.1)

        # 檢驗採樣結果
        if len(self.tags) < 5:
            self.get_logger().error("採樣超時或標籤數據不足，標定失敗！請確保畫面中有指定的 AprilTag。")
            self.calibration_step = 0
            return

        # 3. 多幀姿態求均值 (平滑數據，降低視覺噪聲)
        pose = map(lambda tag: common.xyz_rot_to_mat(tag.pose_t, tag.pose_R), self.tags)
        vectors = map(lambda p: p.ravel(), pose)
        avg_pose = np.mean(list(vectors), axis=0).reshape((4, 4))  # 相機座標系下的 Tag 位姿 (${}^C T_O$)
        
        # 座標系轉換連鎖公式：${}^B T_O = {}^B T_E \cdot {}^E T_C \cdot {}^C T_O$
        pose_end = np.matmul(self.hand2cam_tf_matrix, avg_pose)    # 轉至末端相對座標
        pose_world = np.matmul(endpoint, pose_end)                 # 轉至機械臂基座世界座標系
        
        self.white_area_pose_world = pose_world
        world_position = np.eye(4)
        world_position[:3, 3] = pose_world[:3, 3]
        
        self.white_area_pose_cam = avg_pose
        white_area_pose_cam = avg_pose.tolist()
        white_area_pose_world = world_position.tolist()

        # 4. 使用 OpenCV 的 SolvePnP 計算相機外參 (Extrinsics)
        # 定義 AprilTag 在其自身世界座標下的四個角點真實尺寸 (m)
        world_points = np.array([(-self.tag_size/2, -self.tag_size/2, 0), 
                                 ( self.tag_size/2, -self.tag_size/2, 0), 
                                 ( self.tag_size/2,  self.tag_size/2, 0), 
                                 (-self.tag_size/2,  self.tag_size/2, 0)] * len(self.tags), dtype=np.float64)

        # 影像上的 2D 像素角點
        image_points = np.array(list(map(lambda tag: tag.corners, self.tags)), dtype=np.float64).reshape((-1, 2))
        
        # 利用 PnP 求解外參 (旋轉向量 rvec 與 平移向量 tvec)
        retval, rvec, tvec = cv2.solvePnP(world_points, image_points, self.K, self.D)
        rmat, _ = cv2.Rodrigues(rvec)  # 將旋轉向量轉換為 3x3 旋轉矩陣
        tvec_flattened = tvec.flatten().tolist()

        # 組成 4x3 的外參表達結構
        extristric = [
            tvec_flattened,
            rmat[0].tolist(),
            rmat[1].tolist(),
            rmat[2].tolist()
        ]
        self.extristric = np.array(extristric)
        corners = self.draw_retangle()  # 重建視覺邊框點
        
        # 5. 打包標定結果並自動寫入 YAML 設定檔
        if self.camera_type == "GEMINI":
            data = {
                'white_area_pose_cam': white_area_pose_cam,
                'white_area_pose_world': white_area_pose_world,
                'extristric': extristric,
                'corners': corners.tolist(),
                'plane': self.plane.tolist(),
            }
        else:
            data = {
                'white_area_pose_cam': white_area_pose_cam,
                'white_area_pose_world': white_area_pose_world,
                'extristric': extristric,
                'corners': corners.tolist(),
            }

        self.update_yaml_data(data, self.config_path + self.config_file)
        
        # 發布完成訊號
        msg = Bool()
        msg.data = True
        self.finish_pub.publish(msg)  
        self.calibration_step = 20  # 代表標定成功，畫面將繪製綠色框
        self.get_logger().info(f"=== 手眼標定成功！結果已儲存至：{self.config_path + self.config_file} ===")
        time.sleep(2.0)
        self.running = False  # 讓 image_processing 迴圈結束

    def send_request(self, client, msg):
        """同步等待並發送 ROS 2 Service 請求"""
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def update_yaml_data(self, new_data, yaml_file):
        """將字典資料更新或寫入 YAML 檔案"""
        os.makedirs(os.path.dirname(yaml_file), exist_ok=True)
        if os.path.exists(yaml_file):
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}  

        data.update(new_data)  

        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f)
        time.sleep(0.1)

    def draw_retangle(self):
        """依據標定矩陣計算並投射 3D 工作區域的四個頂點到 2D 像素畫面上"""
        white_area_center = self.white_area_pose_world.reshape(4, 4)
        white_area_cam = self.white_area_pose_cam.reshape(4, 4)

        white_area_lt = np.matmul(white_area_center, common.xyz_euler_to_mat((self.white_area_height / 2, self.white_area_width / 2, 0.0), (0, 0, 0)))
        white_area_lb = np.matmul(white_area_center, common.xyz_euler_to_mat((-self.white_area_height / 2, self.white_area_width / 2, 0.0), (0, 0, 0)))
        white_area_rb = np.matmul(white_area_center, common.xyz_euler_to_mat((-self.white_area_height / 2, -self.white_area_width / 2, 0.0), (0, 0, 0)))
        white_area_rt = np.matmul(white_area_center, common.xyz_euler_to_mat((self.white_area_height / 2, -self.white_area_width / 2, 0.0), (0, 0, 0)))

        endpoint = self.send_request(self.get_current_pose_client, GetRobotPose.Request())
        pose_t = endpoint.pose.position
        pose_r = endpoint.pose.orientation

        endpoint = common.xyz_quat_to_mat([pose_t.x, pose_t.y, pose_t.z], [pose_r.w, pose_r.x, pose_r.y, pose_r.z])
        corners_cam = np.matmul(np.linalg.inv(np.matmul(endpoint, self.hand2cam_tf_matrix)), [white_area_lt, white_area_lb, white_area_rb, white_area_rt, white_area_center])
        corners_cam = np.matmul(np.linalg.inv(white_area_cam), corners_cam)
        corners_cam = corners_cam[:, :3, 3:].reshape((-1, 3))
        tvec = self.extristric[:1]  
        rmat = self.extristric[1:]  

        while self.K is None or self.D is None:
            time.sleep(0.5)

        center_imgpts, _ = cv2.projectPoints(corners_cam[-1:], np.array(rmat), np.array(tvec), self.K, self.D)
        self.center_imgpts = np.int32(center_imgpts).reshape(2)

        tvec, rmat = common.extristric_plane_shift(np.array(tvec).reshape((3, 1)), np.array(rmat), 0.0)
        imgpts, _ = cv2.projectPoints(corners_cam[:-1], np.array(rmat), np.array(tvec), self.K, self.D)

        self.imgpts = np.int32(imgpts).reshape(-1, 2)
        tvec, rmat = common.extristric_plane_shift(np.array(tvec).reshape((3, 1)), np.array(rmat), 0.08)
        imgpts, _ = cv2.projectPoints(corners_cam[:-1], np.array(rmat), np.array(tvec), self.K, self.D)
        self.imgpts1 = np.int32(imgpts).reshape(-1, 2)
        return corners_cam

    def search_plane(self, depth_image):
        """分析深度影像以計算相機前的平面參數"""
        p = self.depth_cam_info.p
        camera_intrinsics = [p[0], p[5], p[2], p[6]]  # [fx, fy, cx, cy]
        searcher = search_plane.SearchPlane(self.depth_cam_info.width, self.depth_cam_info.height, camera_intrinsics)
        a, m, s = searcher.find_plane(depth_image)
        self.plane = m

    def image_processing(self):
        """背景影像處理主迴圈：抓取 Queue 中的影像、偵測 AprilTag 並繪製 UI 狀態疊加畫面"""
        while self.running:
            if self.enter:
                # 從 Queue 中提取影像
                rgb_image = self.image_queue.get(block=True)
                if self.camera_type == 'GEMINI':
                    depth_image = self.depth_image_queue.get(block=True)
                result_image = np.copy(rgb_image)
                
                # 確保相機內參已接收
                if self.K is not None:
                    # 使用 AprilTag Detector 進行 2D 圖像辨識與 3D 姿態估算
                    tags = self.at_detector.detect(
                        cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY), 
                        True, 
                        (self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]), 
                        self.tag_size
                    )
                    result_image = common.draw_tags(result_image, tags)  # 畫出 Tag 的框與坐標軸
                    
                    # 採樣階段 (Step 1)
                    if self.calibration_step == 1:
                        if len(tags) == 1 and (tags[0].tag_id in self.tag_id or tags[0].tag_id == self.tag_id_2):
                            self.err_msg = None
                            if len(self.tags) > 0:
                                # 檢驗標籤連續穩定性：變動小於 3mm 才納入統計
                                if common.distance(self.tags[-1].pose_t, tags[0].pose_t) < 0.003:
                                    self.tags.append(tags[0])
                                else:
                                    self.tags = []
                            else:
                                self.tags.append(tags[0])
                                
                            # 收集滿 10 幀穩定數據
                            if len(self.tags) >= 10:
                                self.get_logger().info("AprilTag 數據收集完成，開始計算幾何平面...")
                                if self.camera_type == 'GEMINI' and self.depth_cam_info is not None:
                                    self.search_plane(depth_image)
                                self.calibration_step = 2  # 推進至計算階段
                        else:
                            self.tags = []
                            if self.err_msg is None:
                                self.err_msg = "請確保視野中只有一個指定 ID 的 AprilTag！"

                # 在影像上印出提示訊息
                if self.err_msg is not None:
                    cv2.putText(result_image, self.err_msg, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if self.calibration_step != 0:
                    msg = "Calibration finished!" if self.calibration_step == 20 else "Calibrating..."
                    cv2.putText(result_image, msg, (5, result_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if self.imgpts is not None:
                    cv2.drawContours(result_image, [self.imgpts], -1, (255, 255, 0), 2, cv2.LINE_AA)

                # 將繪製後的結果發布到 ROS Topic，方便在 RViz2 或 rqt_image_view 查看
                self.result_image_pub.publish(self.bridge.cv2_to_imgmsg(result_image, "rgb8"))
            else:
                time.sleep(0.1)

    # --- ROS 2 Topic Callback 函式 ---
    def camera_info_callback(self, msg):
        """接收彩色相機內參"""
        self.K = np.matrix(msg.k).reshape(1, -1, 3)
        self.D = np.array(msg.d)

    def depth_camera_info_callback(self, msg):
        """接收深度相機內參"""
        self.depth_cam_info = msg

    def image_callback(self, ros_image):
        """將 ROS 影像轉為 OpenCV 格式並丟入 Queue"""
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            self.image_queue.get()
        self.image_queue.put(rgb_image)

    def depth_image_callback(self, ros_depth_image):
        """處理深度影像"""
        depth_image = np.ndarray(shape=(ros_depth_image.height, ros_depth_image.width), dtype=np.uint16, buffer=ros_depth_image.data)
        if self.depth_image_queue.full():
            self.depth_image_queue.get()
        self.depth_image_queue.put(depth_image)


def main():
    """程式入口點"""
    node = StandaloneCalibrationNode('standalone_calibration')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()