import cv2
from train_yolo import YoloDetector
from train_qwen import QwenSemanticBrain
    
def main():
    print("=== 初始化 JetArm AI 系統 ===")
    yolo_eye = YoloDetector('yolo11n.pt') # 之後換成你訓練好的權重
    qwen_brain = QwenSemanticBrain()
    
    # 模擬相機拍到了一張照片 (這裡用讀取圖片代替)
    # frame = cv2.imread('dataset/images/test.jpg') 
    
    # --- 以下為系統運作流程 ---
    print("\n[步驟 1]: 獲取相機即時影像...")
    # 假設這是 YOLO 看完畫面後抓到的東西 (手動模擬資料)
    mock_detected_items = [
        {'name': 'red_block', 'center': (150, 200)},
        {'name': 'blue_block', 'center': (350, 200)},
        {'name': 'green_block', 'center': (150, 400)}
    ]
    
    print(f"\n[步驟 2]: YOLO 發現了 {len(mock_detected_items)} 個物件")
    
    user_input = input("\n請輸入語意指令 (例如：幫我拿第一排右邊那個積木): ")
    
    print("\n[步驟 3]: 啟動 Qwen 進行語意分析與空間解構...")
    target_idx = qwen_brain.get_target_index(user_input, mock_detected_items)
    
    target_object = mock_detected_items[target_idx]
    
    print(f"\n[步驟 4]: 準備夾取！")
    print(f"鎖定目標: {target_object['name']}")
    print(f"像素座標 (u, v): {target_object['center']}")
    print("未來這裡將讀取深度圖的 Z 軸，轉換為實體 (X, Y, Z) 並發送給機械手臂！")

if __name__ == "__main__":
    main()