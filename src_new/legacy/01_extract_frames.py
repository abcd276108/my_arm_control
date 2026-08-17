import os
import cv2

# =========================
# 你要改的參數
# =========================
VIDEO_ROOT = r"video"      # 影片根資料夾，底下有 train / val / test
OUTPUT_ROOT = r"frames"     # 輸出幀圖片的根資料夾
SAVE_EVERY_N_FRAMES = 1     # 每幾幀存 1 張，例如 5 代表每 5 幀取 1 幀
IMAGE_EXT = ".jpg"          # 輸出圖片格式
JPEG_QUALITY = 95           # JPG 品質
# =========================


def ensure_dir(path: str):
    """若資料夾不存在就建立"""
    os.makedirs(path, exist_ok=True)


def get_video_files(folder: str):
    """取得資料夾中所有常見影片檔"""
    valid_exts = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".MP4", ".MOV")
    files = []
    for name in sorted(os.listdir(folder)):
        full_path = os.path.join(folder, name)
        if os.path.isfile(full_path) and name.endswith(valid_exts):
            files.append(full_path)
    return files


def extract_frames_from_video(video_path: str, output_dir: str, save_every_n: int):
    """
    從單支影片切幀
    每 save_every_n 幀存一張
    """
    ensure_dir(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[錯誤] 無法開啟影片：{video_path}")
        return 0, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    saved_count = 0
    frame_idx = 0

    print(f"\n開始處理影片：{video_path}")
    print(f"總幀數：{total_frames} | FPS：{fps:.2f}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每 N 幀存一張
        if frame_idx % save_every_n == 0:
            saved_count += 1
            out_name = f"frame_{saved_count:06d}{IMAGE_EXT}"
            out_path = os.path.join(output_dir, out_name)

            if IMAGE_EXT.lower() in [".jpg", ".jpeg"]:
                cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            else:
                cv2.imwrite(out_path, frame)

        frame_idx += 1

    cap.release()
    print(f"已輸出到：{output_dir}")
    print(f"實際存出張數：{saved_count}")

    return total_frames, saved_count


def main():
    """
    遍歷 videos/train, videos/val, videos/test
    並輸出到 frames/train/video_name, frames/val/video_name, frames/test/video_name
    """
    if not os.path.exists(VIDEO_ROOT):
        print(f"[錯誤] 找不到 VIDEO_ROOT：{VIDEO_ROOT}")
        return

    splits = ["test"]

    for split in splits:
        split_video_dir = os.path.join(VIDEO_ROOT, split)
        if not os.path.exists(split_video_dir):
            print(f"[提示] 沒有這個分組資料夾，跳過：{split_video_dir}")
            continue

        video_files = get_video_files(split_video_dir)
        if not video_files:
            print(f"[提示] {split_video_dir} 裡沒有影片")
            continue

        print(f"\n====================")
        print(f"處理分組：{split}")
        print(f"====================")

        for video_path in video_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(OUTPUT_ROOT, split, video_name)

            extract_frames_from_video(
                video_path=video_path,
                output_dir=output_dir,
                save_every_n=SAVE_EVERY_N_FRAMES
            )

    print("\n全部影片切幀完成。")


if __name__ == "__main__":
    main()