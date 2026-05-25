from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "module4"
VIOLATION_THRESHOLD = 10
RECOVERY_THRESHOLD = 3


@dataclass
class FrameStatus:
    """Store whether one frame is compliant."""

    frame_index: int
    is_violation: bool
    violation_type: str = ""


@dataclass
class AlarmState:
    """State machine for debounced alarm logic."""

    violation_count: int = 0
    recovery_count: int = 0
    alarm_active: bool = False
    last_violation_type: str = ""


@dataclass
class StateTransition:
    """Store a single state transition result."""

    frame_index: int
    is_violation: bool
    violation_count: int
    recovery_count: int
    alarm_active: bool
    triggered: bool
    cleared: bool
    message: str


class AlarmStateMachine:
    """Debounced alarm state machine for helmet compliance checking."""

    def __init__(self, violation_threshold: int = VIOLATION_THRESHOLD, recovery_threshold: int = RECOVERY_THRESHOLD) -> None:
        self.violation_threshold = violation_threshold
        self.recovery_threshold = recovery_threshold
        self.state = AlarmState()

    def update(self, frame_status: FrameStatus) -> StateTransition:
        """Update the alarm state with one frame result."""
        triggered = False
        cleared = False
        message = ""

        if frame_status.is_violation:
            self.state.violation_count += 1
            self.state.recovery_count = 0
            self.state.last_violation_type = frame_status.violation_type or self.state.last_violation_type

            if not self.state.alarm_active and self.state.violation_count >= self.violation_threshold:
                self.state.alarm_active = True
                triggered = True
                message = f"报警触发：连续 {self.state.violation_count} 帧违规，类型={self.state.last_violation_type or 'unknown'}"
            else:
                message = f"违规中：累计 {self.state.violation_count} 帧，暂未达到报警阈值"
        else:
            self.state.recovery_count += 1
            self.state.violation_count = 0

            if self.state.alarm_active and self.state.recovery_count >= self.recovery_threshold:
                self.state.alarm_active = False
                cleared = True
                message = f"报警解除：连续 {self.state.recovery_count} 帧恢复正常"
            else:
                message = f"恢复中：连续正常 {self.state.recovery_count} 帧"

        return StateTransition(
            frame_index=frame_status.frame_index,
            is_violation=frame_status.is_violation,
            violation_count=self.state.violation_count,
            recovery_count=self.state.recovery_count,
            alarm_active=self.state.alarm_active,
            triggered=triggered,
            cleared=cleared,
            message=message,
        )


def build_demo_sequence() -> list[FrameStatus]:
    """Create a demo frame sequence with short noise and a real violation."""
    sequence: list[FrameStatus] = []

    # 前几帧正常。
    for i in range(1, 6):
        sequence.append(FrameStatus(frame_index=i, is_violation=False))

    # 单帧误检，不应立刻报警。
    sequence.append(FrameStatus(frame_index=6, is_violation=True, violation_type="no_helmet"))
    sequence.append(FrameStatus(frame_index=7, is_violation=False))
    sequence.append(FrameStatus(frame_index=8, is_violation=False))

    # 连续违规，应该触发报警。
    for i in range(9, 21):
        sequence.append(FrameStatus(frame_index=i, is_violation=True, violation_type="no_helmet"))

    # 恢复正常，经过几帧后解除报警。
    for i in range(21, 28):
        sequence.append(FrameStatus(frame_index=i, is_violation=False))

    return sequence


def draw_timeline(transitions: list[StateTransition]) -> np.ndarray:
    """Draw a simple timeline for state transitions."""
    width = 1280
    height = 420
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)

    cv2.putText(canvas, "Alarm state machine timeline", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(canvas, f"Violation threshold = {VIOLATION_THRESHOLD}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
    cv2.putText(canvas, f"Recovery threshold = {RECOVERY_THRESHOLD}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

    start_x = 50
    end_x = width - 50
    track_y = 220
    spacing = (end_x - start_x) / max(len(transitions) - 1, 1)

    cv2.line(canvas, (start_x, track_y), (end_x, track_y), (120, 120, 120), 2)

    for idx, transition in enumerate(transitions):
        x = int(start_x + idx * spacing)
        color = (0, 0, 255) if transition.is_violation else (0, 180, 0)
        if transition.alarm_active:
            color = (0, 140, 255)

        # cv2.circle(img, center, radius, color, thickness)
        # thickness=-1 表示填充圆形。
        cv2.circle(canvas, (x, track_y), 10, color, -1)
        cv2.putText(canvas, str(transition.frame_index), (x - 12, track_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)

        if transition.triggered:
            cv2.putText(canvas, "ALARM ON", (x - 30, track_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if transition.cleared:
            cv2.putText(canvas, "ALARM OFF", (x - 30, track_y - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 0), 1)

    legend_y = 320
    cv2.rectangle(canvas, (20, legend_y), (40, legend_y + 20), (0, 180, 0), -1)
    cv2.putText(canvas, "normal", (50, legend_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
    cv2.rectangle(canvas, (150, legend_y), (170, legend_y + 20), (0, 0, 255), -1)
    cv2.putText(canvas, "violation", (180, legend_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
    cv2.rectangle(canvas, (320, legend_y), (340, legend_y + 20), (0, 140, 255), -1)
    cv2.putText(canvas, "alarm active", (350, legend_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)

    return canvas


def main() -> None:
    sequence = build_demo_sequence()
    machine = AlarmStateMachine()
    transitions: list[StateTransition] = []

    print("状态机报警逻辑演示开始")
    print(f"连续违规触发阈值：{VIOLATION_THRESHOLD}")
    print(f"连续恢复解除阈值：{RECOVERY_THRESHOLD}")
    print()

    for frame_status in sequence:
        transition = machine.update(frame_status)
        transitions.append(transition)

        print(
            f"frame={transition.frame_index:02d} | violation={transition.is_violation} | "
            f"v_count={transition.violation_count:02d} | r_count={transition.recovery_count:02d} | "
            f"alarm={transition.alarm_active} | {transition.message}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "03_alarm_state_machine_timeline.jpg"
    canvas = draw_timeline(transitions)
    cv2.imwrite(str(output_path), canvas)

    print()
    print(f"时间线图已保存：{output_path}")

    cv2.imshow("Alarm state machine demo", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
