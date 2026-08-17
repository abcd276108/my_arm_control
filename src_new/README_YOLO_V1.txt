JetArm 單焊槍尖端 YOLO V1
============================

一、建立 YOLO Dataset
---------------------
最簡單：

python build_yolo_dataset.py --overwrite

若資料夾位置不同：

python build_yolo_dataset.py ^
  --dataset-root "D:\JetArm_Project\data\dataset" ^
  --output-dir "D:\JetArm_Project\data\yolo_tip_dataset" ^
  --box-size 24 ^
  --overwrite

重要設定：
- 單一類別：tip
- 依 task 分配 train/val/test，避免同一段連續影格同時進入不同 split。
- has_tip=true：以標註點為中心產生固定大小 bounding box。
- has_tip=false：保留影像並建立空 txt，作為負樣本。
- has_wire：保留在 manifest.csv，但 V1 不拿來分類。
- 預設切分：80% / 10% / 10%。
- 預設 box：24 × 24 px。

輸出：

yolo_tip_dataset/
  images/train, val, test
  labels/train, val, test
  dataset.yaml
  manifest.csv
  build_summary.json

二、開始訓練
------------
先安裝：

pip install ultralytics

訓練：

python train_yolo_tip.py

可調整：

python train_yolo_tip.py --model yolo11n.pt --epochs 100 --imgsz 640 --batch 16 --device 0

最佳權重預設會放在：

runs/welding_tip/yolo_tip_v1/weights/best.pt

三、建議
--------
1. 第一次先使用 box-size 24。
2. 若尖端太細、標註誤差較大，可比較 24、32、40 px。
3. Train/Val/Test 應以 task 切分，不要隨機切單張影格，否則連續畫面太相似，驗證結果會虛高。
4. V1 先只偵測 tip；has_wire 留到後續版本。