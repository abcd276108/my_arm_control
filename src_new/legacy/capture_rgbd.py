import cv2
import numpy as np
import time

def capture_training_images(save_path="dataset/images/", num_images=50):
    """
    用於收集 YOLO 訓練資料的腳本。
    按下 's' 儲存影像，按下 'q' 離開。
    """
    cap = cv2.VideoCapture(0) # 0 為筆電預設鏡頭，未來接上深度相機可改用其 SDK
    print("相機已啟動。按 's' 拍照，按 'q' 退出。")
    
    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            img_name = f"{save_path}/image_{int(time.time())}.jpg"
            cv2.imwrite(img_name, frame)
            # 未來這裡會加入儲存對應 Depth map (.npy) 的程式碼
            print(f"已儲存: {img_name} ({count+1}/{num_images})")
            count += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_training_images()