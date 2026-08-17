import cv2
import os

# 1. 建立一個專屬資料夾來放訓練照片
save_dir = "dataset/images"
os.makedirs(save_dir, exist_ok=True)

# 2. 啟動攝影機 (0 通常是筆電內建鏡頭，如果有外接 USB 鏡頭可以改成 1 或 2 試試看)
cap = cv2.VideoCapture(0)
count = 0

print("📸 鏡頭啟動中...")
print("👉 準備好你要夾取的物件後，按下 's' 鍵拍照")
print("👉 按下 'q' 鍵關閉視窗")

while True:
    # 讀取當前鏡頭畫面
    ret, frame = cap.read()
    if not ret:
        print("❌ 無法讀取畫面，請確認鏡頭是否有插好，或被其他程式佔用！")
        break

    # 顯示即時畫面
    cv2.imshow("Camera View (Press 's' to save, 'q' to quit)", frame)

    # 偵測鍵盤按鍵
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        # 按下 s 儲存照片
        img_name = os.path.join(save_dir, f"sample_{count:03d}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"✅ 成功儲存照片: {img_name}")
        count += 1
        
    elif key == ord('q'):
        # 按下 q 離開迴圈
        print("🛑 結束拍攝。")
        break

# 3. 釋放資源並關閉視窗
cap.release()
cv2.destroyAllWindows()