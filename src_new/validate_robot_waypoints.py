from __future__ import annotations

import argparse
import csv
from pathlib import Path

import rclpy
from rclpy.node import Node

from kinematics import transform
from kinematics.kinematics_control import set_pose_target
from kinematics_msgs.srv import GetRobotPose, SetRobotPose


class WaypointValidator(Node):
    def __init__(self) -> None:
        super().__init__("welding_waypoint_validator")
        self.pose_client = self.create_client(
            GetRobotPose, "/kinematics/get_current_pose"
        )
        self.ik_client = self.create_client(
            SetRobotPose, "/kinematics/set_pose_target"
        )

    def call(self, client, request, timeout: float = 8.0):
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"Service unavailable: {client.srv_name}")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.exception() is not None:
            raise RuntimeError(f"Service call failed: {client.srv_name}")
        return future.result()

    def current_pose(self):
        response = self.call(self.pose_client, GetRobotPose.Request())
        if not response.success or not response.solution:
            raise RuntimeError("Cannot obtain current JetArm pose")
        pitch = float(transform.qua2rpy(response.pose.orientation)[1])
        point = [
            float(response.pose.position.x), float(response.pose.position.y),
            float(response.pose.position.z),
        ]
        return point, pitch

    def solve(self, xyz: list[float], pitch: float, pitch_range: list[float]):
        request = set_pose_target(
            xyz, pitch, pitch_range=pitch_range, resolution=1.0
        )
        return self.call(self.ik_client, request)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate JetArm waypoint IK without publishing servo commands"
    )
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--pulse-margin", type=int, default=20)
    parser.add_argument(
        "--pitch", type=float, default=None,
        help="Override current pitch in degrees; -90 points the tool downward.",
    )
    parser.add_argument("--pitch-range", type=float, default=20.0)
    parser.add_argument(
        "--skip-initial-lift", action="store_true",
        help="Skip the extra 50 mm lift above the current initialized pose.",
    )
    args = parser.parse_args()
    with args.csv.expanduser().open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit("Waypoint CSV is empty")

    rclpy.init()
    node = WaypointValidator()
    try:
        current, current_pitch = node.current_pose()
        pitch = current_pitch if args.pitch is None else args.pitch
        pitch_range = [
            max(-180.0, pitch - abs(args.pitch_range)),
            min(180.0, pitch + abs(args.pitch_range)),
        ]
        print(f"Current initialized pitch: {current_pitch:.4f}")
        print(f"IK target pitch: {pitch:.4f}, search range: {pitch_range}")
        checks: list[tuple[str, list[float], float]] = []
        if not args.skip_initial_lift:
            checks.append((
                "initial safety lift",
                [current[0], current[1], current[2] + 0.05],
                current_pitch,
            ))
        for row in rows:
            down = [
                float(row["robot_x_m"]), float(row["robot_y_m"]),
                float(row["robot_z_m"]),
            ]
            approach = [
                float(row["approach_x_m"]), float(row["approach_y_m"]),
                float(row["approach_z_m"]),
            ] if row.get("approach_x_m") else [down[0], down[1], down[2] + 0.05]
            checks.extend([
                (f"point {row['waypoint_id']} approach", approach, pitch),
                (f"point {row['waypoint_id']} down", down, pitch),
            ])
        accepted = 0
        for label, xyz, target_pitch in checks:
            target_range = (
                [current_pitch - 5.0, current_pitch + 5.0]
                if label == "initial safety lift" else pitch_range
            )
            response = node.solve(xyz, target_pitch, target_range)
            pulses = list(response.pulse) if response is not None else []
            valid = len(pulses) >= 4 and all(
                args.pulse_margin <= int(value) <= 1000 - args.pulse_margin
                for value in pulses[:4]
            )
            print(
                f"{label}: xyz={xyz} "
                f"IK={'OK' if valid else 'FAILED'} pulses={pulses}"
            )
            accepted += int(valid)
        print(f"IK result: {accepted}/{len(checks)} safety checks accepted")
        print("Validation only: no servo command was published.")
        return 0 if accepted == len(checks) else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
