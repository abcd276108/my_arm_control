import cv2
import socket
import struct
import numpy as np
import os
from ultralytics import YOLO

# 設定存放錄影圖片的資料夾
SAVE_DIR = "captured_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

# 載入 YOLO 模型
model = YOLO('yolov8n.pt') 

# 建立網路連線
HOST = '192.168.51.188'
PORT = 9999
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT)) 

recording = False
count = 0
current_depth_map = None  # 用來儲存當前的深度資料庫

# 滑鼠點擊事件：當點擊左邊 RGB 畫面時，去右邊深度圖取值
def get_depth_click(event, x, y, flags, param):
    global current_depth_map
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_depth_map is not None:
            # 因為點擊的是左半邊 RGB，對應到右半邊深度圖的相同位置
            # 我們直接讀取深度圖在 (y, x) 位置的灰階值（或代表距離的數值）
            depth_val = current_depth_map[y, x]
            print(f"📍 點擊 RGB 座標: ({x}, {y}) | 對應深度數值: {depth_val}")

print("🎥 控制說明：按 'r' 開始/停止錄影，按 'q' 離開，在畫面上點擊滑鼠左鍵可測距")

try:
    # 綁定滑鼠點擊事件到即將產生的視窗
    cv2.namedWindow("JetArm Real-time Detection")
    cv2.setMouseCallback("JetArm Real-time Detection", get_depth_click)

    while True:
        # 1. 接收合併圖的資料長度
        raw_len = client_socket.recv(4)
        if not raw_len: break
        data_len = struct.unpack('<L', raw_len)[0]
        
        # 2. 接收合併圖的 bytes 資料
        data = b""
        while len(data) < data_len:
            data += client_socket.recv(data_len - len(data))
        
        # 3. 解碼出合併圖 (這張圖的寬度是 RGB 的兩倍)
        combined_frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if combined_frame is None: continue
        
        h, w, c = combined_frame.shape
        half_w = w // 2  # 找到切分的中線
        
        # 4. 【核心步驟】切割畫面
        rgb_frame = combined_frame[:, :half_w]     # 左半邊：純 RGB 影像
        depth_frame = combined_frame[:, half_w:]   # 右半邊：深度影像
        
        # 將目前的深度圖存入全域變數，供滑鼠點擊查詢
        # 因為深度圖已經轉成 3 頻道灰階，我們取其中一個頻道即可代表深度
        current_depth_map = depth_frame[:, :, 0] 
        
        # 5. 只把「左半邊的 RGB 影像」丟給 YOLO 進行推論
        results = model(rgb_frame, stream=True)
        annotated_frame = rgb_frame.copy()
        for r in results:
            annotated_frame = r.plot()  # 畫上偵測框
        
        # 6. 錄影/儲存資料集 (我們只存乾淨的 RGB 原始畫面，方便以後重新訓練)
        if recording:
            cv2.imwrite(os.path.join(SAVE_DIR, f"cap_{count:06d}.jpg"), rgb_frame)
            count += 1
            cv2.putText(annotated_frame, "RECORDING...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 7. 顯示畫面 (只顯示有 YOLO 框的 RGB 畫面)
        cv2.imshow("JetArm Real-time Detection", annotated_frame)
        
        # 也可以選擇把深度圖也單獨秀出來對照 (選用，不影響 YOLO)
        cv2.imshow("Real Depth Map", depth_frame)

        key = cv2.waitKey(1)
        if key == ord('q'): break
        elif key == ord('r'): 
            recording = not recording
            print(f"錄影狀態: {recording}")

finally:
    client_socket.close()
    cv2.destroyAllWindows()