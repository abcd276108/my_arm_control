from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "RGB + Depth Viewer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="同時查看 RGB 與 16-bit Depth，並顯示滑鼠位置的深度值。"
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--task-dir",
        type=Path,
        help="Task 資料夾，例如 data/dataset/test/task_0001",
    )
    source_group.add_argument(
        "--rgb",
        type=Path,
        help="RGB 圖片路徑；使用此參數時也必須提供 --depth",
    )

    parser.add_argument(
        "--depth",
        type=Path,
        help="Depth PNG 路徑；與 --rgb 一起使用",
    )
    parser.add_argument(
        "--frame",
        default="000005",
        help="使用 --task-dir 時要讀取的影格編號，預設 000005",
    )
    parser.add_argument(
        "--min-depth",
        type=int,
        default=200,
        help="彩色顯示的最小深度，單位 mm，預設 200",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5000,
        help="彩色顯示的最大深度，單位 mm，預設 5000",
    )

    args = parser.parse_args()

    if args.rgb is not None and args.depth is None:
        parser.error("使用 --rgb 時，必須同時提供 --depth。")

    if args.min_depth < 0:
        parser.error("--min-depth 不可小於 0。")

    if args.max_depth <= args.min_depth:
        parser.error("--max-depth 必須大於 --min-depth。")

    return args


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.task_dir is not None:
        frame = Path(args.frame).stem
        rgb_path = args.task_dir / "rgb" / f"{frame}.jpg"
        depth_path = args.task_dir / "depth" / f"{frame}.png"
        return rgb_path, depth_path

    assert args.rgb is not None
    assert args.depth is not None
    return args.rgb, args.depth


def load_images(rgb_path: Path, depth_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not rgb_path.exists():
        raise FileNotFoundError(f"找不到 RGB 圖片：{rgb_path}")

    if not depth_path.exists():
        raise FileNotFoundError(f"找不到 Depth 圖片：{depth_path}")

    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise ValueError(f"RGB 圖片讀取失敗：{rgb_path}")

    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Depth 圖片讀取失敗：{depth_path}")

    if depth.ndim == 3:
        depth = depth[:, :, 0]

    if depth.dtype != np.uint16:
        print(
            f"⚠️ Depth 型別是 {depth.dtype}，不是常見的 uint16。"
            "仍會繼續顯示，但請確認資料格式。"
        )

    if depth.shape[:2] != rgb.shape[:2]:
        print(
            "⚠️ RGB 與 Depth 解析度不同："
            f"RGB={rgb.shape[1]}x{rgb.shape[0]}，"
            f"Depth={depth.shape[1]}x{depth.shape[0]}"
        )
        print("為了顯示與滑鼠查詢，Depth 將以最近鄰方式縮放到 RGB 尺寸。")
        depth = cv2.resize(
            depth,
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    return rgb, depth


def colorize_depth(
    depth: np.ndarray,
    min_depth: int,
    max_depth: int,
) -> np.ndarray:
    clipped = np.clip(depth, min_depth, max_depth)

    normalized = (
        (clipped.astype(np.float32) - min_depth)
        / float(max_depth - min_depth)
        * 255.0
    ).astype(np.uint8)

    # 近距離顯示偏紅，遠距離偏藍。
    normalized = 255 - normalized
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)

    # 深度值 0 通常代表無效量測，以黑色顯示。
    colored[depth == 0] = (0, 0, 0)
    return colored


def add_panel_title(image: np.ndarray, title: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(
        image,
        title,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> int:
    args = parse_args()
    rgb_path, depth_path = resolve_paths(args)
    rgb, depth = load_images(rgb_path, depth_path)

    height, width = rgb.shape[:2]

    mouse_x = 0
    mouse_y = 0

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal mouse_x, mouse_y

        if event == cv2.EVENT_MOUSEMOVE:
            mouse_x = int(np.clip(x % width, 0, width - 1))
            mouse_y = int(np.clip(y, 0, height - 1))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print("========================================")
    print("Depth Viewer")
    print("========================================")
    print(f"RGB   ：{rgb_path}")
    print(f"Depth ：{depth_path}")
    print(f"尺寸  ：{width} x {height}")
    print(f"Depth 型別：{depth.dtype}")
    print(f"有效最小值：{int(depth[depth > 0].min()) if np.any(depth > 0) else 0} mm")
    print(f"最大值：{int(depth.max())} mm")
    print("----------------------------------------")
    print("移動滑鼠：查看像素與深度")
    print("Q / ESC ：結束")
    print("========================================")

    while True:
        rgb_view = rgb.copy()
        depth_view = colorize_depth(depth, args.min_depth, args.max_depth)

        add_panel_title(rgb_view, "RGB")
        add_panel_title(
            depth_view,
            f"Depth color map ({args.min_depth}-{args.max_depth} mm)",
        )

        depth_mm = int(depth[mouse_y, mouse_x])
        depth_text = "INVALID" if depth_mm == 0 else f"{depth_mm} mm"

        for panel in (rgb_view, depth_view):
            cv2.drawMarker(
                panel,
                (mouse_x, mouse_y),
                (255, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
            )
            cv2.circle(panel, (mouse_x, mouse_y), 6, (0, 0, 0), 2)

            info = f"Pixel=({mouse_x}, {mouse_y})  Depth={depth_text}"
            cv2.rectangle(
                panel,
                (0, height - 38),
                (width, height),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                panel,
                info,
                (10, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        combined = np.hstack((rgb_view, depth_view))
        cv2.imshow(WINDOW_NAME, combined)

        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())