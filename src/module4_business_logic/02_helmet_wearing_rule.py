from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "module4"


@dataclass
class FaceKeypoints:
    """Simple face keypoints for rule-based helmet checking."""

    left_eye: tuple[int, int]
    right_eye: tuple[int, int]
    nose: tuple[int, int]
    chin: tuple[int, int]
    left_ear: tuple[int, int]
    right_ear: tuple[int, int]


@dataclass
class HelmetBox:
    """Helmet bounding box in xyxy format."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def bottom_y(self) -> int:
        return self.y2


@dataclass
class HelmetRuleResult:
    """Store the result of rule-based helmet checking."""

    helmet_present: bool
    correct_position: bool
    chinstrap_ok: bool
    passed: bool
    reasons: list[str]


def make_demo_face() -> FaceKeypoints:
    """Create a simple demo face layout."""
    return FaceKeypoints(
        left_eye=(220, 180),
        right_eye=(280, 180),
        nose=(250, 220),
        chin=(250, 290),
        left_ear=(190, 210),
        right_ear=(310, 210),
    )


def make_demo_helmet() -> HelmetBox:
    """Create a helmet box that is roughly in the right place."""
    return HelmetBox(x1=190, y1=110, x2=310, y2=210)


def evaluate_helmet_position(helmet: HelmetBox, face: FaceKeypoints) -> tuple[bool, list[str]]:
    """Check whether the helmet sits above the eyes and near the head center."""
    reasons = []
    correct_position = True

    eye_line_y = (face.left_eye[1] + face.right_eye[1]) / 2.0
    face_center_x = (face.left_eye[0] + face.right_eye[0]) / 2.0
    helmet_center_x, _ = helmet.center

    # 安全帽下边缘应该高于眼睛，至少有一点安全空间。
    if helmet.bottom_y >= eye_line_y - 5:
        correct_position = False
        reasons.append("helmet too low relative to eyes")

    # 安全帽中心应该大致对准脸部中心，太偏左或偏右就可能是戴歪了。
    face_width = face.right_eye[0] - face.left_eye[0]
    if abs(helmet_center_x - face_center_x) > face_width * 0.6:
        correct_position = False
        reasons.append("helmet center too far from face center")

    return correct_position, reasons


def evaluate_chinstrap(face: FaceKeypoints, helmet: HelmetBox) -> tuple[bool, list[str]]:
    """Check a simple chinstrap proxy rule using chin and ear geometry."""
    reasons = []
    chinstrap_ok = True

    helmet_center_x, _ = helmet.center
    chin_x, chin_y = face.chin
    ear_mid_y = (face.left_ear[1] + face.right_ear[1]) / 2.0

    # 这里用简化规则：帽体中心要和下颚区域、耳朵中线保持一定合理性。
    # 真正工程里通常会结合颜色、关键点和二次分类器判断帽带是否经过下颚。
    if abs(helmet_center_x - chin_x) > helmet.width * 0.35:
        chinstrap_ok = False
        reasons.append("helmet center not aligned with chin")

    if helmet.bottom_y < ear_mid_y - 20:
        chinstrap_ok = False
        reasons.append("helmet too high for chinstrap region")

    return chinstrap_ok, reasons


def evaluate_helmet_rule(helmet: HelmetBox | None, face: FaceKeypoints) -> HelmetRuleResult:
    """Evaluate a rule-based helmet wearing check."""
    reasons: list[str] = []

    if helmet is None:
        return HelmetRuleResult(
            helmet_present=False,
            correct_position=False,
            chinstrap_ok=False,
            passed=False,
            reasons=["no helmet detected"],
        )

    helmet_present = True
    correct_position, position_reasons = evaluate_helmet_position(helmet, face)
    chinstrap_ok, chinstrap_reasons = evaluate_chinstrap(face, helmet)

    reasons.extend(position_reasons)
    reasons.extend(chinstrap_reasons)

    passed = helmet_present and correct_position and chinstrap_ok
    return HelmetRuleResult(
        helmet_present=helmet_present,
        correct_position=correct_position,
        chinstrap_ok=chinstrap_ok,
        passed=passed,
        reasons=reasons,
    )


def draw_demo(face: FaceKeypoints, helmet: HelmetBox | None, result: HelmetRuleResult) -> np.ndarray:
    """Draw face keypoints, helmet box, and rule result on a canvas."""
    canvas = np.full((420, 500, 3), 245, dtype=np.uint8)

    # 画出脸部关键点，帮助理解规则几何关系。
    for point, color, label in [
        (face.left_eye, (255, 0, 0), "L-eye"),
        (face.right_eye, (255, 0, 0), "R-eye"),
        (face.nose, (0, 128, 255), "nose"),
        (face.chin, (0, 0, 255), "chin"),
        (face.left_ear, (128, 0, 255), "L-ear"),
        (face.right_ear, (128, 0, 255), "R-ear"),
    ]:
        cv2.circle(canvas, point, 6, color, -1)
        cv2.putText(canvas, label, (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    if helmet is not None:
        # cv2.rectangle(img, pt1, pt2, color, thickness)
        # pt1/pt2: 左上角和右下角坐标。
        # color: BGR 颜色。
        # thickness=2: 边框粗细。
        color = (0, 200, 0) if result.passed else (0, 0, 255)
        cv2.rectangle(canvas, (helmet.x1, helmet.y1), (helmet.x2, helmet.y2), color, 2)
        cv2.putText(canvas, "helmet", (helmet.x1, max(20, helmet.y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    status_color = (0, 160, 0) if result.passed else (0, 0, 255)
    status_text = "CORRECT HELMET" if result.passed else "INCORRECT HELMET"
    cv2.putText(canvas, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

    lines = [
        f"helmet_present: {result.helmet_present}",
        f"correct_position: {result.correct_position}",
        f"chinstrap_ok: {result.chinstrap_ok}",
    ]

    for idx, line in enumerate(lines, start=1):
        cv2.putText(canvas, line, (20, 330 + idx * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)

    if result.reasons:
        cv2.putText(canvas, "reasons:", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
        for idx, reason in enumerate(result.reasons[:3], start=1):
            cv2.putText(canvas, f"- {reason}", (20, 240 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

    return canvas


def main() -> None:
    face = make_demo_face()
    helmet = make_demo_helmet()
    result = evaluate_helmet_rule(helmet, face)

    print("安全帽佩戴关键点规则演示")
    print(f"helmet_present = {result.helmet_present}")
    print(f"correct_position = {result.correct_position}")
    print(f"chinstrap_ok = {result.chinstrap_ok}")
    print(f"passed = {result.passed}")
    if result.reasons:
        print("原因：")
        for reason in result.reasons:
            print(f"- {reason}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "02_helmet_wearing_rule_demo.jpg"
    canvas = draw_demo(face, helmet, result)
    cv2.imwrite(str(output_path), canvas)

    print(f"结果图已保存：{output_path}")

    cv2.imshow("Helmet wearing rule demo", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
