from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path='yolo11n.pt'):
        # 初始化 YOLOv11 模型 (n 為輕量版，訓練後請改成你自己的 best.pt)
        self.model = YOLO(model_path)
        
    def train_model(self, data_yaml_path='../dataset/data.yaml', epochs=100):
        """使用 4090 進行極速訓練"""
        print("開始訓練 YOLOv11...")
        self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=16,
            device=0 # 強制使用 RTX 4090
        )
        
    def detect_objects(self, image_frame):
        """輸入影像，回傳物件清單與 2D 座標框"""
        results = self.model(image_frame)
        detected_items = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 取得 bounding box 座標 (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # 計算中心點 (u, v)
                center_u = int((x1 + x2) / 2)
                center_v = int((y1 + y2) / 2)
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id]
                
                detected_items.append({
                    'name': class_name,
                    'center': (center_u, center_v),
                    'box': (x1, y1, x2, y2)
                })
        return detected_items

if __name__ == "__main__":
    # 單獨執行此檔案時，啟動訓練模式
    detector = YoloDetector()
    # detector.train_model() # 準備好 data.yaml 後把這行取消註解