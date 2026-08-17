from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shlex
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import cv2
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from main_control import DatasetManager, RecordingSession, decode_packet

try:
    from workboard_calibration import CalibrationError, pixel_to_board, pixel_to_robot, validate_config
    CALIBRATION_IMPORT_ERROR = ""
except ImportError as exc:
    CALIBRATION_IMPORT_ERROR = str(exc)

    class CalibrationError(RuntimeError):
        pass

    def _calibration_unavailable(*_args: object, **_kwargs: object):
        raise CalibrationError(f"Workboard calibration module unavailable: {CALIBRATION_IMPORT_ERROR}")

    pixel_to_board = _calibration_unavailable
    pixel_to_robot = _calibration_unavailable
    validate_config = _calibration_unavailable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = PROJECT_ROOT / "hmi"
RUNS_ROOT = PROJECT_ROOT / "runs"
CALIBRATION_PATH = PROJECT_ROOT / "config" / "workboard_calibration.json"
EXECUTION_LOCK = threading.Lock()
EXECUTION_STATUS: dict[str, object] = {
    "state": "idle", "message": "等待操作員確認", "run": "",
}


def start_robot_execution(run_name: str) -> dict[str, object]:
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("Invalid fusion run")
    csv_path = (
        RUNS_ROOT / "infer_yolo_depth_fusion" / run_name
        / "trajectory" / "robot_waypoints.csv"
    )
    if not csv_path.is_file():
        raise FileNotFoundError(f"Robot waypoint CSV not found: {csv_path}")
    with EXECUTION_LOCK:
        if EXECUTION_STATUS.get("state") == "running":
            raise RuntimeError("Robot trajectory is already running")
        EXECUTION_STATUS.update({
            "state": "running",
            "message": "手臂執行中，請勿進入工作區",
            "run": run_name,
            "started_at": time.time(),
            "output": "",
        })

    def worker() -> None:
        # The competition dry-run uses a 3 mm DOWN clearance and keeps a
        # 30 mm vertical retract between points.  Re-export immediately before
        # execution so the HMI can never run an older 5/10/15 mm waypoint CSV.
        calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        calibration.setdefault("heights_mm", {})
        calibration["heights_mm"]["down_dry_run_mm"] = 5
        calibration["heights_mm"]["up_safe_mm"] = 35
        CALIBRATION_PATH.write_text(
            json.dumps(calibration, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        fusion_csv = csv_path.parent.parent / "fusion_results.csv"
        export_command = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "export_workboard_waypoints.py"),
            "--fusion", str(fusion_csv),
            "--calibration", str(CALIBRATION_PATH),
            "--output", str(csv_path),
            "--merge-gap-seconds", "0.50",
            "--min-segment-frames", "4",
            "--split-up-frames", "3",
            "--confidence", "0.25",
        ]
        command = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "execute_robot_waypoints.py"),
            "--csv", str(csv_path), "--all", "--execute", "--yes",
            "--pitch", "60", "--pitch-range", "30",
            "--pulse-margin", "50", "--travel-duration", "5",
            "--step-duration", "3", "--down-dwell", "5",
            "--initial-lift-mm", "30",
            "--transition-pause", "0.2",
        ]
        try:
            exported = subprocess.run(
                export_command, cwd=PROJECT_ROOT, text=True, encoding="utf-8",
                errors="replace", capture_output=True, timeout=180,
            )
            if exported.returncode != 0:
                raise RuntimeError(
                    "Failed to rebuild 5 mm robot waypoints: "
                    + (exported.stderr or exported.stdout).strip()
                )
            result = subprocess.run(
                command, cwd=PROJECT_ROOT, text=True, encoding="utf-8",
                errors="replace", capture_output=True, timeout=600,
            )
            output = f"{result.stdout}\n{result.stderr}".strip()
            with EXECUTION_LOCK:
                EXECUTION_STATUS.update({
                    "state": "completed" if result.returncode == 0 else "failed",
                    "message": (
                        "實機軌跡執行完成"
                        if result.returncode == 0 else "實機軌跡執行失敗"
                    ),
                    "output": output[-12000:],
                    "finished_at": time.time(),
                })
        except Exception as exc:
            with EXECUTION_LOCK:
                EXECUTION_STATUS.update({
                    "state": "failed", "message": str(exc),
                    "finished_at": time.time(),
                })

    threading.Thread(target=worker, name="robot-trajectory", daemon=True).start()
    return dict(EXECUTION_STATUS)


class CameraStreamClient:
    """Receives synchronized packets from ros2_stream_bridge and exposes JPEG frames."""

    HEADER_SIZE = struct.calcsize("<L")
    MAX_PACKET_SIZE = 128 * 1024 * 1024

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.latest_sequence = -1
        self.latest_received_at = 0.0
        self.latest_packet: dict[str, object] | None = None
        self.last_error = "waiting for camera bridge"
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="camera-stream-client", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    @staticmethod
    def _recv_exact(connection: socket.socket, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = connection.recv(size - len(chunks))
            if not chunk:
                raise ConnectionError("camera bridge disconnected")
            chunks.extend(chunk)
        return bytes(chunks)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                with socket.create_connection((self.host, self.port), timeout=3.0) as connection:
                    connection.settimeout(5.0)
                    with self.lock:
                        self.last_error = ""
                    while not self.stop_event.is_set():
                        packet_size = struct.unpack("<L", self._recv_exact(connection, self.HEADER_SIZE))[0]
                        if packet_size <= 0 or packet_size > self.MAX_PACKET_SIZE:
                            raise ValueError(f"invalid camera packet size: {packet_size}")
                        packet = pickle.loads(self._recv_exact(connection, packet_size))
                        jpeg = packet.get("rgb") if isinstance(packet, dict) else None
                        if not isinstance(jpeg, (bytes, bytearray)) or not jpeg:
                            continue
                        with self.lock:
                            self.latest_jpeg = bytes(jpeg)
                            self.latest_packet = packet
                            self.latest_sequence = int(packet.get("sequence", self.latest_sequence + 1))
                            self.latest_received_at = time.monotonic()
                        record_camera_packet(packet)
            except Exception as exc:
                with self.lock:
                    self.last_error = str(exc)
                self.stop_event.wait(1.0)

    def snapshot(self) -> tuple[bytes | None, int, float, str]:
        with self.lock:
            age = time.monotonic() - self.latest_received_at if self.latest_received_at else float("inf")
            return self.latest_jpeg, self.latest_sequence, age, self.last_error

    def ready(self) -> bool:
        jpeg, _sequence, age, _error = self.snapshot()
        return jpeg is not None and age < 3.0

    def packet_snapshot(self) -> dict[str, object] | None:
        with self.lock:
            return self.latest_packet


CAMERA_CLIENT = CameraStreamClient(
    os.environ.get("JETARM_CAMERA_BRIDGE_HOST", "127.0.0.1"),
    int(os.environ.get("JETARM_CAMERA_BRIDGE_PORT", "9999")),
)

RECORDING_LOCK = threading.Lock()
RECORDING_MANAGER = DatasetManager(PROJECT_ROOT / "data" / "dataset")
RECORDING_SESSION: RecordingSession | None = None


def record_camera_packet(packet: dict[str, object]) -> None:
    with RECORDING_LOCK:
        session = RECORDING_SESSION
        if session is None or not session.is_recording:
            return
        rgb, depth = decode_packet(packet)
        session.process_frame(rgb, depth, packet)


def start_recording(category: str = "runtime_capture") -> dict[str, object]:
    global RECORDING_SESSION
    safe_category = "".join(c for c in category if c.isalnum() or c in "_-").strip("_-") or "runtime_capture"
    packet = CAMERA_CLIENT.packet_snapshot()
    if packet is None or not CAMERA_CLIENT.ready():
        raise RuntimeError("即時 RGB-D 相機尚未就緒")
    with RECORDING_LOCK:
        if RECORDING_SESSION is not None and RECORDING_SESSION.is_recording:
            raise RuntimeError("目前已有示教正在錄製")
        category_dir = RECORDING_MANAGER.dataset_root / safe_category
        category_dir.mkdir(parents=True, exist_ok=True)
        session = RecordingSession(RECORDING_MANAGER, category_dir, sample_interval=1, fps=30.0)
        rgb, _depth = decode_packet(packet)
        height, width = rgb.shape[:2]
        session.start((width, height))
        RECORDING_SESSION = session
        return {
            "success": True,
            "task": str(session.task_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "task_name": session.task_name,
        }


def stop_recording(status: str = "完成") -> dict[str, object]:
    global RECORDING_SESSION
    with RECORDING_LOCK:
        session = RECORDING_SESSION
        if session is None or not session.is_recording:
            raise RuntimeError("目前沒有正在錄製的示教")
        session.stop("已取消" if status == "cancelled" else "完成")
        task_dir = session.task_dir
        result = {
            "success": True,
            "task": str(task_dir.relative_to(PROJECT_ROOT)).replace("\\", "/") if task_dir else "",
            "task_name": session.task_name,
            "received_frames": session.received_frame_id,
            "rgb_count": session.rgb_count,
            "depth_count": session.depth_count,
            "duration_seconds": round(
                (session.end_dt - session.start_dt).total_seconds(), 2
                if session.end_dt and session.start_dt else 0
            ) if session.end_dt and session.start_dt else 0,
        }
        RECORDING_SESSION = None
        return result


def read_calibration() -> dict[str, object]:
    if not CALIBRATION_PATH.is_file():
        return {"calibration_status": "empty"}
    return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))


def save_calibration(payload: dict[str, object]) -> dict[str, object]:
    config = validate_config(payload)
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CALIBRATION_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(CALIBRATION_PATH)
    return config


def project_file(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    path.relative_to(PROJECT_ROOT)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def run_script(*arguments: str) -> None:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, *arguments], cwd=PROJECT_ROOT, text=True,
        encoding="utf-8", errors="replace", capture_output=True, timeout=300,
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "處理失敗")


def create_browser_preview(run_dir: Path) -> bool:
    clean_source = run_dir / "clean_inference.mp4"
    if (run_dir / "fusion_results.csv").is_file():
        try:
            run_script(
                str(PROJECT_ROOT / "src/render_clean_inference_video.py"),
                "--fusion", str(run_dir / "fusion_results.csv"),
                "--output", str(clean_source),
            )
        except Exception as exc:
            print(f"Clean preview unavailable: {exc}")
    source = clean_source if clean_source.is_file() else run_dir / "fusion_evaluation.mp4"
    output = run_dir / "fusion_preview.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if not source.is_file() or ffmpeg is None:
        return False
    result = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(source),
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-an", str(output)],
        capture_output=True, text=True, timeout=300,
    )
    return result.returncode == 0 and output.is_file()


def export_trajectory(predictions_value: str) -> dict[str, object]:
    predictions = project_file(predictions_value)
    output_dir = predictions.parent / "trajectory"
    run_script(str(PROJECT_ROOT / "src/export_trajectory.py"), "--predictions", str(predictions), "--output-dir", str(output_dir))
    trajectory = output_dir / "trajectory.csv"
    waypoints = output_dir / "gazebo_waypoints.csv"
    run_script(str(PROJECT_ROOT / "src/export_gazebo_waypoints.py"), "--trajectory", str(trajectory), "--output", str(waypoints), "--states", "DOWN", "--one-per-segment")
    return {
        "trajectory": str(trajectory.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "waypoints": str(waypoints.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "trajectory_summary": json.loads((output_dir / "trajectory_summary.json").read_text(encoding="utf-8")),
        "waypoint_summary": json.loads(waypoints.with_suffix(".summary.json").read_text(encoding="utf-8")),
    }


def run_fusion(task_value: str, model_value: str) -> dict[str, object]:
    task = (PROJECT_ROOT / task_value).resolve()
    model = project_file(model_value)
    task.relative_to(PROJECT_ROOT)
    if not (task / "rgb").is_dir() or not (task / "depth").is_dir():
        raise FileNotFoundError(f"Task 缺少 rgb 或 depth：{task}")
    output_root = RUNS_ROOT / "infer_yolo_depth_fusion"
    before = set(output_root.glob("*")) if output_root.exists() else set()
    run_script(
        str(PROJECT_ROOT / "src/infer_yolo_depth_fusion.py"),
        "--task-dir", str(task), "--model", str(model),
        "--output-root", str(output_root), "--no-show", "--save-video",
    )
    candidates = [path for path in output_root.glob("*") if path.is_dir() and path not in before]
    if not candidates:
        candidates = [path for path in output_root.glob("*") if path.is_dir()]
    run_dir = max(candidates, key=lambda path: path.stat().st_mtime)
    fusion_csv = run_dir / "fusion_results.csv"
    trajectory = run_dir / "trajectory" / "trajectory.csv"
    waypoints = run_dir / "trajectory" / "gazebo_waypoints.csv"
    run_script(str(PROJECT_ROOT / "src/export_fusion_xyz.py"), "--fusion", str(fusion_csv), "--output", str(trajectory))
    run_script(
        str(PROJECT_ROOT / "src/export_gazebo_waypoints.py"),
        "--trajectory", str(trajectory), "--output", str(waypoints),
        "--states", "DOWN", "--one-per-segment",
    )
    robot_waypoints = run_dir / "trajectory" / "robot_waypoints.csv"
    gazebo_down_points = run_dir / "trajectory" / "gazebo_down_points.csv"
    try:
        calibration = read_calibration()
        if calibration.get("calibration_status") == "ready":
            run_script(
                str(PROJECT_ROOT / "src/export_workboard_waypoints.py"),
                "--fusion", str(fusion_csv),
                "--calibration", str(CALIBRATION_PATH),
                "--output", str(robot_waypoints),
                "--merge-gap-seconds", "0.50",
                "--min-segment-frames", "4",
                "--split-up-frames", "3",
                "--confidence", "0.25",
            )
    except Exception as exc:
        print(f"Workboard waypoint export unavailable: {exc}")
    package_dir = PROJECT_ROOT / "handoff" / "simulation_queue" / run_dir.name
    package_dir.mkdir(parents=True, exist_ok=True)
    package_files = [
        fusion_csv,
        run_dir / "fusion_summary.json",
        trajectory,
        trajectory.with_suffix(".summary.json"),
        waypoints,
        waypoints.with_suffix(".summary.json"),
        robot_waypoints,
        robot_waypoints.with_suffix(".summary.json"),
        gazebo_down_points,
    ]
    for source in package_files:
        if source.is_file():
            shutil.copy2(source, package_dir / source.name)
    waypoint_summary = json.loads(waypoints.with_suffix(".summary.json").read_text(encoding="utf-8"))
    preview_ready = create_browser_preview(run_dir)
    manifest = {
        "schema_version": 1,
        "task_id": run_dir.name,
        "source_task": str(task.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "coordinate_frame": waypoint_summary.get("coordinate_frame", "camera_relative"),
        "unit": "meter",
        "waypoint_file": "gazebo_waypoints.csv",
        "status": "pending_simulation",
        "robot_execution_ready": bool(waypoint_summary.get("robot_execution_ready", False)),
        "note": "Camera-relative trajectory for Gazebo path validation; not a robot-base trajectory.",
    }
    (package_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    archive_path = shutil.make_archive(str(package_dir), "zip", root_dir=package_dir)
    return {
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "fusion_summary": json.loads((run_dir / "fusion_summary.json").read_text(encoding="utf-8")),
        "trajectory_summary": json.loads(trajectory.with_suffix(".summary.json").read_text(encoding="utf-8")),
        "waypoint_summary": waypoint_summary,
        "preview_ready": preview_ready,
        "simulation_package": str(package_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "simulation_archive": str(Path(archive_path).relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, payload: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, payload: object, status: int = 200) -> None:
        self.send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/api/fusion-down-points", "/api/fusion-down-frame"}:
            try:
                query = parse_qs(parsed.query)
                run_name = query.get("run", [""])[0]
                if not run_name or Path(run_name).name != run_name:
                    raise ValueError("Invalid fusion run")
                run_dir = RUNS_ROOT / "infer_yolo_depth_fusion" / run_name
                waypoint_path = run_dir / "trajectory" / "robot_waypoints.csv"
                with waypoint_path.open("r", encoding="utf-8-sig", newline="") as handle:
                    points = list(csv.DictReader(handle))
                if path == "/api/fusion-down-points":
                    self.send_json({"count": len(points)})
                    return
                index = int(query.get("index", ["0"])[0])
                if index < 0 or index >= len(points):
                    raise IndexError("Down point index out of range")
                point = points[index]
                frame_id = str(point.get("source_frame_id", ""))
                with (run_dir / "fusion_results.csv").open(
                    "r", encoding="utf-8-sig", newline=""
                ) as handle:
                    frame_row = next(
                        (row for row in csv.DictReader(handle) if str(row.get("frame_id", "")) == frame_id),
                        None,
                    )
                if frame_row is None:
                    raise FileNotFoundError(f"Source frame not found: {frame_id}")
                image_path = Path(str(frame_row.get("rgb_path", ""))).resolve()
                image_path.relative_to(PROJECT_ROOT)
                frame = cv2.imread(str(image_path))
                if frame is None:
                    raise FileNotFoundError(image_path)
                x = int(round(float(point["pixel_x"])))
                y = int(round(float(point["pixel_y"])))
                cv2.circle(frame, (x, y), 7, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 5, (53, 63, 156), -1, cv2.LINE_AA)
                cv2.putText(
                    frame, f"DOWN {index + 1}/{len(points)}", (18, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, (53, 63, 156), 2, cv2.LINE_AA,
                )
                encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not encoded:
                    raise RuntimeError("Unable to encode down point preview")
                self.send_bytes(jpeg.tobytes(), "image/jpeg")
            except (FileNotFoundError, IndexError, KeyError, TypeError, ValueError) as exc:
                self.send_json({"detail": str(exc)}, 404)
            except Exception as exc:
                self.send_json({"detail": str(exc)}, 500)
            return
        if path == "/api/fusion-video":
            run_name = parse_qs(parsed.query).get("run", [""])[0]
            if not run_name or Path(run_name).name != run_name:
                self.send_json({"detail": "Invalid fusion run"}, 400)
                return
            candidate = RUNS_ROOT / "infer_yolo_depth_fusion" / run_name / "fusion_preview.mp4"
            if not candidate.is_file():
                self.send_json({"detail": "Fusion preview video not found"}, 404)
                return
            self.send_bytes(candidate.read_bytes(), "video/mp4")
            return
        if path == "/api/fusion-preview":
            run_name = parse_qs(parsed.query).get("run", [""])[0]
            if not run_name or Path(run_name).name != run_name:
                self.send_json({"detail": "Invalid fusion run"}, 400)
                return
            candidate = RUNS_ROOT / "infer_yolo_depth_fusion" / run_name / "fusion_evaluation.mp4"
            capture = cv2.VideoCapture(str(candidate))
            if not candidate.is_file() or not capture.isOpened():
                capture.release()
                self.send_json({"detail": "Fusion preview video not found"}, 404)
                return
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            self.send_response(200)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    if not encoded:
                        continue
                    payload = jpeg.tobytes()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(payload + b"\r\n")
                    self.wfile.flush()
                    time.sleep(1.0 / max(1.0, fps))
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            finally:
                capture.release()
            return
        if path == "/api/tasks/latest-runtime":
            tasks = sorted(
                (PROJECT_ROOT / "data/dataset/runtime_capture").glob("task_*"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            if not tasks:
                self.send_json({"task": ""})
                return
            task = tasks[0]
            metadata_path = task / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
            received = int(metadata.get("received_frame_count", 0))
            fps = float(metadata.get("recording_fps", 30.0)) or 30.0
            self.send_json({
                "success": True,
                "task": str(task.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "task_name": task.name,
                "rgb_count": int(metadata.get("rgb_count", 0)),
                "depth_count": int(metadata.get("depth_count", 0)),
                "duration_seconds": round(received / fps, 2),
            })
            return
        if path == "/api/predictions":
            items = sorted(RUNS_ROOT.glob("infer_yolo/**/predictions.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            self.send_json({"items": [str(p.relative_to(PROJECT_ROOT)).replace("\\", "/") for p in items]})
            return
        if path == "/api/tasks":
            items = sorted((PROJECT_ROOT / "data/dataset").glob("*/task_*"))
            self.send_json({"items": [str(p.relative_to(PROJECT_ROOT)).replace("\\", "/") for p in items if (p / "rgb").is_dir() and (p / "depth").is_dir()]})
            return
        if path == "/api/models":
            items = sorted(RUNS_ROOT.glob("smart_teaching_tip/**/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            self.send_json({"items": [str(p.relative_to(PROJECT_ROOT)).replace("\\", "/") for p in items]})
            return
        if path == "/api/robot/saved-trajectories":
            saved_root = PROJECT_ROOT / "data" / "saved_trajectories"
            records: list[dict[str, object]] = []
            if saved_root.is_dir():
                for folder in saved_root.iterdir():
                    if not folder.is_dir():
                        continue
                    metadata_path = folder / "metadata.json"
                    try:
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    except Exception:
                        metadata = {}
                    records.append({
                        "name": metadata.get("name") or folder.name,
                        "source_run": metadata.get("source_run") or "",
                        "saved_at": metadata.get("saved_at") or "",
                        "path": str(folder.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    })
            records.sort(key=lambda item: str(item.get("saved_at", "")), reverse=True)
            self.send_json({"items": records})
            return
        if path == "/api/health":
            arm_enabled = os.environ.get("WELDING_ARM_ENABLE", "0") == "1"
            initialize_command = os.environ.get(
                "WELDING_INITIALIZE_COMMAND", "ros2 run my_arm_control initialize"
            ).strip()
            self.send_json({
                "status": "ready",
                "camera_stream": "ready" if CAMERA_CLIENT.ready() else "waiting",
                "inference": "offline_only",
                "gazebo_bridge": "not_configured",
                "physical_execution": arm_enabled and (
                    PROJECT_ROOT / "src" / "execute_robot_waypoints.py"
                ).is_file(),
                "robot_initializer": "ready" if arm_enabled and bool(initialize_command) else "locked",
                "workboard_calibration": "ready" if not CALIBRATION_IMPORT_ERROR else "unavailable",
            })
            return
        if path == "/api/robot/execution-status":
            with EXECUTION_LOCK:
                self.send_json(dict(EXECUTION_STATUS))
            return
        if path == "/api/camera/frame.jpg":
            jpeg, _sequence, _age, error = CAMERA_CLIENT.snapshot()
            if jpeg is None:
                self.send_json({"detail": error or "Camera frame unavailable"}, 503)
            else:
                self.send_bytes(jpeg, "image/jpeg")
            return
        if path == "/api/camera/stream":
            self.send_response(200)
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last_sequence = -1
            try:
                while True:
                    jpeg, sequence, age, _error = CAMERA_CLIENT.snapshot()
                    if jpeg is not None and age < 3.0 and sequence != last_sequence:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                        last_sequence = sequence
                    time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            return
        if path == "/api/calibration":
            self.send_json(read_calibration())
            return
        assets = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/runtime": ("runtime.html", "text/html; charset=utf-8"),
            "/runtime.css": ("runtime.css", "text/css; charset=utf-8"),
            "/runtime-init.css": ("runtime-init.css", "text/css; charset=utf-8"),
            "/runtime.js": ("runtime.js", "text/javascript; charset=utf-8"),
            "/calibration": ("calibration.html", "text/html; charset=utf-8"),
            "/calibration.css": ("calibration.css", "text/css; charset=utf-8"),
            "/calibration.js": ("calibration.js", "text/javascript; charset=utf-8"),
        }
        if path in assets:
            name, content_type = assets[path]
            self.send_bytes((WEB_ROOT / name).read_bytes(), content_type)
            return
        self.send_json({"detail": "Not found"}, 404)

    def do_POST(self) -> None:
        endpoint = urlparse(self.path).path
        if endpoint in {"/api/calibration", "/api/calibration/preview"}:
            try:
                size = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(size).decode("utf-8"))
                if endpoint == "/api/calibration":
                    config = save_calibration(body)
                    self.send_json({"success": True, "config": config})
                else:
                    config = body.get("config") or read_calibration()
                    pixel = body.get("pixel")
                    state = str(body.get("state", "DOWN"))
                    board = pixel_to_board(config, pixel)
                    response: dict[str, object] = {
                        "board_xy_mm": [round(float(value), 3) for value in board],
                        "calibration_status": config.get("calibration_status", "empty"),
                    }
                    try:
                        robot = pixel_to_robot(config, pixel, state)
                        response["robot_xyz_mm"] = [round(float(value), 3) for value in robot]
                    except CalibrationError:
                        response["robot_xyz_mm"] = None
                    self.send_json(response)
            except (CalibrationError, KeyError, TypeError, ValueError) as exc:
                self.send_json({"detail": str(exc)}, 400)
            except Exception as exc:
                self.send_json({"detail": str(exc)}, 500)
            return
        if endpoint == "/api/robot/initialize":
            if os.environ.get("WELDING_ARM_ENABLE", "0") != "1":
                self.send_json({"detail": "Robot movement is safety-locked"}, 409)
                return
            try:
                initialize_command = os.environ.get(
                    "WELDING_INITIALIZE_COMMAND", "ros2 run my_arm_control initialize"
                ).strip()
                command = shlex.split(initialize_command)
                working_directory = PROJECT_ROOT
                result = subprocess.run(
                    command, cwd=working_directory,
                    text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=30,
                )
                output = f"{result.stdout}\n{result.stderr}".strip()
                success = result.returncode == 0
                self.send_json({"success": success, "message": output}, 200 if success else 500)
            except Exception as exc:
                self.send_json({"detail": str(exc)}, 500)
            return
        if endpoint == "/api/robot/save-trajectory":
            try:
                size = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(size).decode("utf-8")) if size else {}
                run_name = str(body.get("run", "")).strip()
                if not run_name or Path(run_name).name != run_name:
                    raise ValueError("Invalid fusion run")
                display_name = str(body.get("name", "")).strip() or f"trajectory_{run_name}"
                safe_name = "".join(
                    char if char.isalnum() or char in "-_" else "_" for char in display_name
                ).strip("_")[:80] or f"trajectory_{run_name}"
                source = RUNS_ROOT / "infer_yolo_depth_fusion" / run_name / "trajectory"
                robot_csv = source / "robot_waypoints.csv"
                if not robot_csv.is_file():
                    raise FileNotFoundError(f"Robot waypoint CSV not found: {robot_csv}")
                target = PROJECT_ROOT / "data" / "saved_trajectories" / safe_name
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(robot_csv, target / robot_csv.name)
                gazebo_csv = source / "gazebo_down_points.csv"
                if gazebo_csv.is_file():
                    shutil.copy2(gazebo_csv, target / gazebo_csv.name)
                metadata = {
                    "name": display_name,
                    "source_run": run_name,
                    "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                (target / "metadata.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                self.send_json({
                    "success": True,
                    "path": str(target.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                })
            except (FileNotFoundError, ValueError) as exc:
                self.send_json({"detail": str(exc)}, 400)
            except Exception as exc:
                self.send_json({"detail": str(exc)}, 500)
            return
        if endpoint == "/api/robot/execute-trajectory":
            if os.environ.get("WELDING_ARM_ENABLE", "0") != "1":
                self.send_json({"detail": "Robot movement is safety-locked"}, 409)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(size).decode("utf-8")) if size else {}
                if body.get("operator_confirmed") is not True:
                    self.send_json({"detail": "Operator confirmation is required"}, 400)
                    return
                self.send_json(start_robot_execution(str(body.get("run", ""))), 202)
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                self.send_json({"detail": str(exc)}, 409)
            except Exception as exc:
                self.send_json({"detail": str(exc)}, 500)
            return
        if endpoint in {"/api/recording/start", "/api/recording/stop"}:
            try:
                size = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(size).decode("utf-8")) if size else {}
                if endpoint.endswith("/start"):
                    self.send_json(start_recording(str(body.get("category", "runtime_capture"))))
                else:
                    self.send_json(stop_recording(str(body.get("status", "completed"))))
            except Exception as exc:
                self.send_json({"detail": str(exc)}, 500)
            return
        if endpoint not in {"/api/export", "/api/fusion"}:
            self.send_json({"detail": "Not found"}, 404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(size).decode("utf-8"))
            if endpoint == "/api/fusion":
                self.send_json(run_fusion(str(body.get("task", "")), str(body.get("model", ""))))
            else:
                self.send_json(export_trajectory(str(body.get("predictions", ""))))
        except Exception as exc:
            self.send_json({"detail": str(exc)}, 500)

    def log_message(self, format: str, *args: object) -> None:
        print(f"HMI {self.address_string()}: {format % args}")


def main() -> int:
    parser = argparse.ArgumentParser(description="JetArm 本機人機介面")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    CAMERA_CLIENT.start()
    print(f"JetArm HMI: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        CAMERA_CLIENT.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
