from __future__ import annotations

import argparse
import csv
import math
import re
import time
from pathlib import Path

import rclpy
from rclpy.node import Node

from kinematics.kinematics_control import set_pose_target
from kinematics_msgs.srv import GetRobotPose, SetRobotPose
from servo_controller_msgs.msg import ServoPosition, ServosPosition, ServoStateList


class SafeWaypointExecutor(Node):
    def __init__(self) -> None:
        super().__init__("welding_safe_waypoint_executor")
        self.ik_client = self.create_client(
            SetRobotPose, "/kinematics/set_pose_target"
        )
        self.smooth_ik_client = self.create_client(
            SetRobotPose, "/kinematics/set_pose_target_smooth"
        )
        self.pose_client = self.create_client(
            GetRobotPose, "/kinematics/get_current_pose"
        )
        self.servo_pub = self.create_publisher(
            ServosPosition, "/servo_controller", 10
        )
        self.current_positions: dict[int, int] = {}
        self.state_sub = self.create_subscription(
            ServoStateList, "/controller_manager/servo_states",
            self._on_servo_states, 10,
        )

    def _on_servo_states(self, message: ServoStateList) -> None:
        self.current_positions = {
            int(state.id): int(state.position) for state in message.servo_state
        }

    def solve(
        self, xyz: list[float], pitch: float, pitch_range: list[float]
    ) -> list[int]:
        if not self.ik_client.wait_for_service(timeout_sec=8.0):
            raise RuntimeError("IK service unavailable: /kinematics/set_pose_target")
        request = set_pose_target(
            xyz, pitch, pitch_range=pitch_range, resolution=1.0
        )
        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
        if not future.done() or future.exception() is not None:
            raise RuntimeError("IK service call failed")
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError(f"No IK solution for xyz={xyz}")
        return [int(value) for value in response.pulse]

    def solve_smooth(
        self,
        xyz: list[float],
        pitch: float,
        pitch_range: list[float],
        duration: float,
    ) -> list[list[int]]:
        """Return the quintic Cartesian/IK trajectory supplied by kinematics.

        The smooth service flattens its ``[frame][five servo pulses]`` result
        into ``response.pulse``.  Keep the conversion here so the execution
        path can publish every frame rather than jumping directly to the last
        IK solution.
        """
        if not self.smooth_ik_client.wait_for_service(timeout_sec=8.0):
            raise RuntimeError(
                "Smooth IK service unavailable: /kinematics/set_pose_target_smooth"
            )
        request = set_pose_target(
            xyz, pitch, pitch_range=pitch_range, resolution=1.0,
            duration=float(duration),
        )
        future = self.smooth_ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        if not future.done() or future.exception() is not None:
            raise RuntimeError("Smooth IK service call failed")
        response = future.result()
        raw_pulses = [] if response is None else list(response.pulse)
        if response is None or not response.success or not raw_pulses:
            raise RuntimeError(f"No smooth IK solution for xyz={xyz}")
        if len(raw_pulses) % 5:
            raise RuntimeError(
                f"Smooth IK returned {len(raw_pulses)} pulses, not whole 5-axis frames"
            )
        return [
            [int(value) for value in raw_pulses[index:index + 5]]
            for index in range(0, len(raw_pulses), 5)
        ]

    def current_pose(self) -> tuple[list[float], float]:
        if not self.pose_client.wait_for_service(timeout_sec=8.0):
            raise RuntimeError("Current-pose service unavailable")
        future = self.pose_client.call_async(GetRobotPose.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
        response = future.result() if future.done() else None
        if response is None or not response.success or not response.solution:
            raise RuntimeError("Unable to read current tool pose")
        pose = response.pose
        q = pose.orientation
        value = max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x)))
        pitch = math.degrees(math.asin(value))
        return [pose.position.x, pose.position.y, pose.position.z], pitch

    def publish_pulses(self, pulses: list[int], duration: float) -> None:
        if len(pulses) < 5:
            raise RuntimeError(f"Expected 5 servo pulses, got {pulses}")
        deadline = time.monotonic() + 5.0
        while self.servo_pub.get_subscription_count() == 0:
            if time.monotonic() >= deadline:
                raise RuntimeError("No subscriber on /servo_controller")
            rclpy.spin_once(self, timeout_sec=0.1)
        message = ServosPosition()
        message.duration = float(duration)
        message.position_unit = "pulse"
        # Motor 10 is intentionally never included.
        for servo_id, pulse in zip((1, 2, 3, 4, 5), pulses[:5]):
            position = ServoPosition()
            position.id = servo_id
            position.position = float(pulse)
            message.position.append(position)
        self.servo_pub.publish(message)
        target = {servo_id: int(pulse) for servo_id, pulse in zip((1, 2, 3, 4, 5), pulses[:5])}
        deadline = time.monotonic() + float(duration) + 6.0
        stable_samples = 0
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            reached = self.current_positions and all(
                abs(self.current_positions.get(servo_id, -10000) - pulse) <= 3
                for servo_id, pulse in target.items()
            )
            stable_samples = stable_samples + 1 if reached else 0
            if stable_samples >= 3:
                print(f"Reached pulses={pulses[:5]}", flush=True)
                return
        raise RuntimeError(
            f"Servo target timeout; target={target}, actual={self.current_positions}"
        )

    def publish_smooth_pose(
        self,
        xyz: list[float],
        pitch: float,
        pitch_range: list[float],
        duration: float,
        frame_duration: float,
        pulse_margin: int,
    ) -> None:
        """Plan from the live pose and stream its IK frames at a fixed rate."""
        frames = self.solve_smooth(xyz, pitch, pitch_range, duration)
        for frame in frames:
            if any(pulse < pulse_margin or pulse > 1000 - pulse_margin for pulse in frame[:4]):
                raise RuntimeError(f"Unsafe pulse margin in smooth trajectory: {frame}")

        deadline = time.monotonic() + 5.0
        while self.servo_pub.get_subscription_count() == 0:
            if time.monotonic() >= deadline:
                raise RuntimeError("No subscriber on /servo_controller")
            rclpy.spin_once(self, timeout_sec=0.1)

        frame_duration = max(0.01, float(frame_duration))
        next_send = time.monotonic()
        for frame in frames:
            self._publish_frame(frame, frame_duration)
            next_send += frame_duration
            remaining = next_send - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

        # Do not start the next segment until the last streamed frame is
        # actually reached; this also makes the next smooth IK start pose true.
        self._wait_for_pulses(frames[-1], timeout=6.0)

    def _publish_frame(self, pulses: list[int], duration: float) -> None:
        message = ServosPosition()
        message.duration = float(duration)
        message.position_unit = "pulse"
        for servo_id, pulse in zip((1, 2, 3, 4, 5), pulses[:5]):
            position = ServoPosition()
            position.id = servo_id
            position.position = float(pulse)
            message.position.append(position)
        self.servo_pub.publish(message)

    def _wait_for_pulses(self, pulses: list[int], timeout: float) -> None:
        target = {servo_id: int(pulse) for servo_id, pulse in zip((1, 2, 3, 4, 5), pulses[:5])}
        deadline = time.monotonic() + timeout
        stable_samples = 0
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            reached = self.current_positions and all(
                abs(self.current_positions.get(servo_id, -10000) - pulse) <= 3
                for servo_id, pulse in target.items()
            )
            stable_samples = stable_samples + 1 if reached else 0
            if stable_samples >= 3:
                print(f"Reached smooth endpoint pulses={pulses[:5]}", flush=True)
                return
        raise RuntimeError(
            f"Smooth trajectory endpoint timeout; target={target}, actual={self.current_positions}"
        )


def interpolate(approach: list[float], down: list[float], fraction: float) -> list[float]:
    return [
        approach[index] + fraction * (down[index] - approach[index])
        for index in range(3)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Slow, confirmed JetArm dry-run execution of workboard waypoints"
    )
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--point", type=int, default=0)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--yes", action="store_true",
        help="Use only after an external HMI operator confirmation.",
    )
    parser.add_argument("--pitch", type=float, default=60.0)
    parser.add_argument("--pitch-range", type=float, default=30.0)
    parser.add_argument("--pulse-margin", type=int, default=50)
    parser.add_argument("--travel-duration", type=float, default=5.0)
    parser.add_argument("--step-duration", type=float, default=3.0)
    parser.add_argument(
        "--trajectory-mode", choices=("smooth", "point-to-point"), default="smooth",
        help=("Use the kinematics quintic-interpolation service (default), or "
              "the previous one-command-per-waypoint behavior."),
    )
    parser.add_argument(
        "--smooth-frame-duration", type=float, default=0.02,
        help="Seconds between smooth IK frames; must match the kinematics service rate.",
    )
    parser.add_argument("--down-dwell", type=float, default=5.0)
    parser.add_argument("--transition-pause", type=float, default=0.2)
    parser.add_argument("--initial-lift-mm", type=float, default=30.0)
    parser.add_argument("--lift-step-mm", type=float, default=5.0)
    args = parser.parse_args()

    with args.csv.expanduser().open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit("Waypoint CSV is empty")
    selected = rows if args.all else [rows[args.point]]

    rclpy.init()
    node = SafeWaypointExecutor()
    try:
        pitch_range = [
            args.pitch - abs(args.pitch_range),
            args.pitch + abs(args.pitch_range),
        ]
        current_xyz, current_pitch = node.current_pose()
        first_row = selected[0]
        lift_step = max(1.0, args.lift_step_mm) / 1000.0
        # The first action is always an exact relative vertical lift. No
        # horizontal or downward target is allowed before it completes.
        relative_lift_z = current_xyz[2] + max(0.0, args.initial_lift_mm) / 1000.0
        configured_approach_z = float(first_row["approach_z_m"])
        # Lift by at least 30 mm. If the current pose starts unusually low,
        # continue strictly upward until the configured travel plane is safe.
        lift_target_z = max(relative_lift_z, configured_approach_z)
        lift_levels = []
        lift_z = current_xyz[2] + lift_step
        while lift_z < lift_target_z - 1e-9:
            lift_levels.append(lift_z)
            lift_z += lift_step
        lift_levels.append(lift_target_z)
        lift_pitch_range = [current_pitch - 15.0, current_pitch + 15.0]
        initial_lift = []
        achieved_lift_z = current_xyz[2]
        for lift_z in lift_levels:
            xyz = [current_xyz[0], current_xyz[1], lift_z]
            pulses = node.solve(xyz, current_pitch, lift_pitch_range)
            valid = len(pulses) >= 5 and all(
                args.pulse_margin <= pulse <= 1000 - args.pulse_margin
                for pulse in pulses[:4]
            )
            print(
                f"initial vertical lift: xyz={xyz} pulses={pulses} "
                f"safety={'OK' if valid else 'FAILED'}"
            )
            if not valid:
                achieved_mm = (achieved_lift_z - current_xyz[2]) * 1000.0
                if achieved_mm + 0.75 < args.initial_lift_mm:
                    raise RuntimeError(
                        f"Initial lift reached only {achieved_mm:.1f} mm before "
                        "the pulse safety limit; no motion published"
                    )
                print(
                    f"Initial lift capped at safe height after {achieved_mm:.1f} mm; "
                    "the remaining rise will occur vertically above point 0"
                )
                break
            initial_lift.append((xyz, pulses, args.step_duration))
            achieved_lift_z = lift_z

        lift_target_z = achieved_lift_z

        first_approach = [
            float(first_row["approach_x_m"]),
            float(first_row["approach_y_m"]),
            lift_target_z,
        ]
        clearance_pulses = node.solve(first_approach, args.pitch, pitch_range)
        clearance_valid = len(clearance_pulses) >= 5 and all(
            args.pulse_margin <= pulse <= 1000 - args.pulse_margin
            for pulse in clearance_pulses[:4]
        )
        print(
            f"first-point safe horizontal travel: xyz={first_approach} "
            f"pulses={clearance_pulses} "
            f"safety={'OK' if clearance_valid else 'FAILED'}"
        )
        if not clearance_valid:
            raise RuntimeError("Unsafe first-point high travel clearance")

        plans = []
        for row in selected:
            down = [
                float(row["robot_x_m"]),
                float(row["robot_y_m"]),
                float(row["robot_z_m"]),
            ]
            approach = [
                float(row["approach_x_m"]),
                float(row["approach_y_m"]),
                float(row["approach_z_m"]),
            ]
            vertical_span = math.dist(approach, down)
            if vertical_span < 0.015:
                raise RuntimeError(
                    f"Unsafe height span at point {row['waypoint_id']}: {vertical_span:.4f} m"
                )
            if not math.isclose(
                vertical_span * 1000.0,
                args.initial_lift_mm,
                abs_tol=0.75,
            ):
                raise RuntimeError(
                    f"Point {row['waypoint_id']} lift span is "
                    f"{vertical_span * 1000.0:.1f} mm, expected "
                    f"{args.initial_lift_mm:.1f} mm"
                )
            mode_match = re.search(r"([0-9]+(?:\.[0-9]+)?)mm", str(row.get("height_mode", "")))
            down_height = float(mode_match.group(1)) if mode_match else 0.0
            up_height = down_height + vertical_span * 1000.0
            down_label = str(row.get("height_mode") or "DOWN")
            target_heights = [up_height]
            next_height = math.floor((up_height - 0.001) / 10.0) * 10.0
            while next_height > down_height:
                target_heights.append(next_height)
                next_height -= 10.0
            target_heights.append(down_height)
            levels = []
            for index, height in enumerate(target_heights):
                fraction = (up_height - height) / (vertical_span * 1000.0)
                xyz = approach if index == 0 else (
                    down if index == len(target_heights) - 1
                    else interpolate(approach, down, fraction)
                )
                label = down_label if index == len(target_heights) - 1 else f"{height:g} mm"
                duration = args.travel_duration if index == 0 else args.step_duration
                levels.append((label, xyz, duration))
            solved = []
            for label, xyz, duration in levels:
                pulses = node.solve(xyz, args.pitch, pitch_range)
                valid = len(pulses) >= 5 and all(
                    args.pulse_margin <= pulse <= 1000 - args.pulse_margin
                    for pulse in pulses[:4]
                )
                print(
                    f"point {row['waypoint_id']} {label}: xyz={xyz} "
                    f"pulses={pulses} safety={'OK' if valid else 'FAILED'}"
                )
                if not valid:
                    raise RuntimeError(
                        f"Unsafe pulse margin at point {row['waypoint_id']} {label}"
                    )
                solved.append((label, xyz, pulses, duration))
            plans.append((row, solved))

        if not args.execute:
            print("Plan only: no servo command was published. Add --execute to move.")
            return 0

        if not args.yes:
            answer = input("Type LIFT to begin with a vertical safety lift: ").strip()
            if answer != "LIFT":
                raise RuntimeError("Execution cancelled before initial lift")
        print("Beginning vertical safety lift; XY and tool pitch are held constant")
        for xyz, pulses, duration in initial_lift:
            if args.trajectory_mode == "smooth":
                node.publish_smooth_pose(
                    xyz, current_pitch, lift_pitch_range, duration,
                    args.smooth_frame_duration, args.pulse_margin,
                )
            else:
                node.publish_pulses(pulses, duration)
            # The endpoint check in either publication path has completed;
            # retain only the configured short pause before the next lift.
            time.sleep(args.transition_pause)
        if not args.yes:
            first_id = first_row["waypoint_id"]
            answer = input(
                f"Type TRAVEL {first_id} to move horizontally above point {first_id}: "
            ).strip()
            if answer != f"TRAVEL {first_id}":
                raise RuntimeError("Execution cancelled before horizontal travel")
        print("Moving horizontally above the first point at high clearance")
        if args.trajectory_mode == "smooth":
            node.publish_smooth_pose(
                first_approach, args.pitch, pitch_range, args.travel_duration,
                args.smooth_frame_duration, args.pulse_margin,
            )
        else:
            node.publish_pulses(clearance_pulses, args.travel_duration)
        time.sleep(args.transition_pause)

        for row, solved in plans:
            point_id = row["waypoint_id"]
            if not args.yes:
                answer = input(
                    f"Type MOVE {point_id} to execute this dry-run point: "
                ).strip()
                if answer != f"MOVE {point_id}":
                    raise RuntimeError("Execution cancelled by operator")
            for index, (label, xyz, pulses, duration) in enumerate(solved):
                print(f"Moving point {point_id} to {label} in {duration:.1f}s")
                if args.trajectory_mode == "smooth":
                    node.publish_smooth_pose(
                        xyz, args.pitch, pitch_range, duration,
                        args.smooth_frame_duration, args.pulse_margin,
                    )
                else:
                    node.publish_pulses(pulses, duration)
                time.sleep(args.transition_pause)
                if index == len(solved) - 1:
                    print(
                        f"Point {point_id} reached DOWN; "
                        f"dwelling for {args.down_dwell:.1f}s"
                    )
                    time.sleep(args.down_dwell)
            for label, xyz, pulses, duration in reversed(solved[:-1]):
                print(f"Retracting point {point_id} to {label} in {duration:.1f}s")
                if args.trajectory_mode == "smooth":
                    node.publish_smooth_pose(
                        xyz, args.pitch, pitch_range, duration,
                        args.smooth_frame_duration, args.pulse_margin,
                    )
                else:
                    node.publish_pulses(pulses, duration)
                time.sleep(args.transition_pause)
            print(f"Point {point_id} dry-run complete at {solved[0][0]} safe height")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
