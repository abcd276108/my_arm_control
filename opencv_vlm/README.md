# OpenCV + VLM PCB 焊接流程（概念驗證）

本資料夾將粗略語意判讀交給本機 Qwen2.5-VL，將格點與腳位座標交給 OpenCV。它支援機械手臂上固定的 **640×480 相機**：原始影像會先透視校正到共同的板面座標，再進行推論。它仍是**離線／模擬流程**；沒有可直接控制真實機械臂的程式，也不會呼叫雲端或 HTTP API。

## 資料流程

1. `hole_location.py` 讀取 640×480 空板照片，依 `config/camera_profile.json` 校正為俯視影像並輸出 `grid_map.json`。
2. `unified_component_inference.py` 是唯一的元件／腳位推論入口：同視角會由 Qwen 找元件後用 OpenCV 精定位腳位；跨視角則由 Qwen 回傳 pin 的 `row/col`，並從格點表查精確像素。
3. `vlm_solder_planner.py` 產生語意焊接順序；`path_planner.py` 再驗證每個目標格點並生成軌跡。
4. `arm_execution_and_aoi.py` 只會模擬動作。AOI 影像、端點或回覆失敗時會標為 `UNKNOWN` 並停止，不會假設 PASS。

## 安裝與校正

```powershell
cd opencv_vlm
python -m pip install -r requirements.txt
python hole_location.py
```

所有原始照片／影片放在 `input/`，所有產生檔案放在 `output/`。預設空板檔名為 `input/empty_board.jpg`；可在 `project_settings.py` 修改。不要將 `board_rectified.jpg` 放到 `input/`，因為它是校正流程產生的 `output/board_rectified.jpg`。

第一次執行 `hole_location.py` 時，若沒有 `input/`，程式會自動建立並提示你匯入原始空板照片與待測照片／影片。空板照片應命名為 `empty_board.jpg`，之後再重新執行校正。

上述指令會建立：

- `output/board_rectified.jpg`：校正後影像
- `output/grid_result.jpg`：供人工檢查的格點疊圖；綠色為量測合格、紅色為殘差過大、橘色為僅模型預測。
- `output/grid_map.json`：後續程式使用的格點資料格式
- `output/grid_measurements.json`：每個格點的預測、實測中心與殘差
- `output/calibration_report.json`：匹配率、平均／RMS／最大殘差與 PASS/FAIL 判定

`config/camera_profile.json` 是固定相機的設定檔。其預設四角只適用於目前範例的 640×480 影像；請在相機固定、PCB 夾具固定後，將 `board_corners_px` 改為實測值。程式會拒絕不是 640×480 的縮放或裁切影像，因為該影像不能沿用既有標定。
`path_planner.py` 會自動拒絕同目錄中標為 `FAIL` 的校正報告。`--allow-failed-calibration` 只供離線除錯，禁止用於實機。

## 命令列範例

```powershell
# 用手眼平面標定檔將已審核的 welding_plan.json 轉為軌跡。
python path_planner.py welding_plan.json output/grid_map.json --robot-calibration output/robot_camera_calibration.json

# 只進行模擬；未設定 AOI 時會安全地輸出 UNKNOWN 並停止。
python arm_execution_and_aoi.py output/robot_trajectory.json --aoi-dir aoi_images --model-path D:\models\Qwen2.5-VL-7B-Instruct
```

### 統一推論入口

`unified_component_inference.py` 是唯一保留的元件／腳位推論程式。來自固定機械臂相機的 640×480 原圖會自動校正後採同視角流程；不同角度近照請明確指定 `--mode cross-view`。

`project_settings.py` 是唯一的路徑設定檔，可集中設定 `DEFAULT_MODEL_PATH`、輸入檔名與資料夾。設定後可不帶任何參數直接執行；命令列參數仍會覆寫預設值。`DEFAULT_REFERENCE_IMAGE` 已固定指向校正流程產生的 `output/board_rectified.jpg`。

```powershell
# 同視角照片：Qwen 找 bbox，再由 OpenCV 差分定位兩端腳位。
python opencv_vlm/unified_component_inference.py output/board_rectified.jpg current_rectified.jpg output/grid_map.json --model-path D:\models\Qwen2.5-VL-7B-Instruct

# 不同角度近照：Qwen 判斷 row/col，再由 grid_map 查精確像素。
python opencv_vlm/unified_component_inference.py output/board_rectified.jpg component_closeup.jpg output/grid_map.json --mode cross-view --model-path D:\models\Qwen2.5-VL-7B-Instruct

# 固定相機影片：第一幀為空板基準、最後一幀為目前畫面；可用 --base-image 改用既有空板圖。
python opencv_vlm/unified_component_inference.py output/board_rectified.jpg insertion.mp4 output/grid_map.json --model-path D:\models\Qwen2.5-VL-7B-Instruct

# 已在程式頂端設定好預設路徑時，可直接執行。
python opencv_vlm/unified_component_inference.py
```

## 機械手臂相機的首次設定

1. 將相機與 PCB／夾具完全固定；拍攝一張無元件的 `input/empty_board.jpg`，解析度必須是 640×480。
2. 在圖片工具中量出板面四角的像素座標（左上、右上、右下、左下），填入 `config/camera_profile.json` 的 `board_corners_px`。同時確認格點數、`rectified_size` 與邊界值。
3. 執行 `python hole_location.py`，確認 `output/grid_result.jpg` 與 `output/calibration_report.json` 為 `PASS`。此步會保存實際使用的 profile 到 `output/camera_profile_used.json`。
4. 將探針／治具依序移到至少 6 個分散的板面孔位；記錄每個孔在「校正後板面影像」中的 `pixel:[x,y]`，以及對應機械臂 base 座標 `arm_xy_mm:[X,Y]`。存成 `input/robot_correspondences.json`，例如：

```json
[
  {"pixel": [10.2, 9.4], "arm_xy_mm": [120.14, 88.02]},
  {"pixel": [330.1, 190.3], "arm_xy_mm": [168.22, 115.27]}
]
```

5. 以實測點建立手眼平面標定（至少 6 點，建議 9 點以上且分散在整塊板上）：

```powershell
python robot_calibration.py input/robot_correspondences.json --surface-z 10.0 --workspace 80 220 40 180
```

輸出的 `output/robot_camera_calibration.json` 必須是 `PASS`（RMS ≤ 1 mm、最大誤差 ≤ 2 mm）才可被 `path_planner.py` 使用。這個轉換直接使用每一孔的實測像素座標，不再使用示範用的「每格固定毫米」假設。

影片只支援固定相機的同視角推論；程式會將使用的首尾影格輸出在 JSON 相同資料夾，以便核對。跨角度影片請先擷取欲分析的近照，再以 `--mode cross-view` 執行。

`--model-path` 必須指向已完整下載至本機的 Qwen2.5-VL-Instruct 模型資料夾。程式以 `local_files_only=True` 載入模型，因此即使電腦沒有網路也不會下載權重或呼叫 API。官方 Qwen 範例採用 Transformers 的 `Qwen2_5_VLForConditionalGeneration`、`AutoProcessor` 與 `qwen-vl-utils`；本專案依此在本機直接推論。[Qwen 官方範例](https://github.com/QwenLM/Qwen2.5-VL) 

跨視角映射會自動產生帶 `row,col` 標籤的空板參考圖給模型。Qwen 回傳的每個 row/col 都必須存在於 `grid_map.json`，最終像素一律採用該表的實測孔洞中心；不存在、重複或越界的格點會直接拒絕輸出。校正報告為 `FAIL` 時也會拒絕執行。

## 進入實機前的必要條件

- 以標定板量測相機校正、格點誤差與格點到機械臂座標的轉換誤差。
- 在 `path_planner.py` 設定實際工作區、Z 安全高度及經驗證的手眼校正矩陣。
- 實作獨立且經安全審核的機械臂 adapter：急停、碰撞／工作區檢查、速度限制、焊槍互鎖與人工確認皆不可省略。
- 蒐集 AOI 的真實標註資料並量化 false-pass rate；模型或相機異常必須保持 fail-closed。
