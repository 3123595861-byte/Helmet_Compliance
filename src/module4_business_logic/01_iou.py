from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "module4"
IOU_THRESHOLD = 0.5


@dataclass
class Box:
    """Axis-aligned bounding box in xyxy format."""

    x1: float
    y1: float
    x2: float
    y2: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class IoUTestCase:
    """Store a pair of boxes for IoU demo."""

    name: str
    box_a: Box
    box_b: Box


@dataclass
class IoUResult:
    """Store the result of one IoU test."""

    name: str
    box_a: Box
    box_b: Box
    iou: float
    passed: bool


def compute_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Compute IoU between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def compare_boxes(box_a: Box, box_b: Box, threshold: float = IOU_THRESHOLD) -> IoUResult:
    """Compare two boxes and decide whether they match."""
    iou = compute_iou(box_a.as_tuple(), box_b.as_tuple())
    return IoUResult(
        name="pair",
        box_a=box_a,
        box_b=box_b,
        iou=iou,
        passed=iou >= threshold,
    )


def get_demo_cases() -> list[IoUTestCase]:
    """Create three cases: overlap, no overlap, and partial overlap."""
    return [
        IoUTestCase(
            name="high_overlap",
            box_a=Box(100, 100, 220, 220),
            box_b=Box(110, 110, 230, 230),
        ),
        IoUTestCase(
            name="no_overlap",
            box_a=Box(100, 100, 180, 180),
            box_b=Box(220, 220, 300, 300),
        ),
        IoUTestCase(
            name="partial_overlap",
            box_a=Box(100, 100, 220, 220),
            box_b=Box(160, 160, 280, 280),
        ),
    ]


def draw_box(canvas: np.ndarray, box: Box, color: tuple[int, int, int], label: str) -> None:
    """Draw one box and its label on a canvas."""
    x1, y1, x2, y2 = map(int, box.as_tuple())

    # cv2.rectangle(img, pt1, pt2, color, thickness)
    # pt1/pt2: 左上角和右下角坐标，格式是 (x, y)。
    # color: OpenCV 使用 BGR。
    # thickness=2: 线条粗细。
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    cv2.putText(canvas, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def build_canvas(case: IoUTestCase, result: IoUResult) -> np.ndarray:
    """Create a simple visualization for one IoU case."""
    canvas = np.full((360, 420, 3), 245, dtype=np.uint8)

    draw_box(canvas, case.box_a, (0, 0, 255), "A")
    draw_box(canvas, case.box_b, (255, 0, 0), "B")

    status = "PASS" if result.passed else "FAIL"
    text = f"IoU={result.iou:.3f}  threshold={IOU_THRESHOLD:.2f}  {status}"
    cv2.putText(canvas, text, (20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 0), 2)
    cv2.putText(canvas, case.name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 0), 2)

    return canvas


def helmet_head_demo() -> tuple[Box, Box]:
    """Create a simple helmet/head rule demo."""
    head_box = Box(140, 120, 240, 240)
    helmet_box = Box(145, 110, 245, 210)
    return helmet_box, head_box


def main() -> None:
    print("IoU 计算演示开始")
    print(f"判定阈值：{IOU_THRESHOLD}")
    print("IoU = 交集面积 / 并集面积")
    print()

    cases = get_demo_cases()
    canvases = []

    for case in cases:
        result = compare_boxes(case.box_a, case.box_b, IOU_THRESHOLD)
        canvases.append(build_canvas(case, result))

        print(f"Case: {case.name}")
        print(f"  box_a = {case.box_a.as_tuple()}, area={case.box_a.area:.1f}")
        print(f"  box_b = {case.box_b.as_tuple()}, area={case.box_b.area:.1f}")
        print(f"  IoU   = {result.iou:.4f}")
        print(f"  判定  = {'正确佩戴/匹配' if result.passed else '不匹配/不正确'}")
        print()

    helmet_box, head_box = helmet_head_demo()
    helmet_head_iou = compute_iou(helmet_box.as_tuple(), head_box.as_tuple())
    helmet_correct = helmet_head_iou >= IOU_THRESHOLD

    print("安全帽框 vs 头框示例")
    print(f"  helmet_box = {helmet_box.as_tuple()}")
    print(f"  head_box   = {head_box.as_tuple()}")
    print(f"  IoU        = {helmet_head_iou:.4f}")
    print(f"  判定       = {'疑似正确佩戴' if helmet_correct else '未正确佩戴'}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for case, canvas in zip(cases, canvases):
        output_path = OUTPUT_DIR / f"01_iou_{case.name}.jpg"
        cv2.imwrite(str(output_path), canvas)
        output_paths.append(output_path)

    print()
    for path in output_paths:
        print(f"结果图已保存：{path}")

    comparison = np.vstack([cv2.imread(str(path)) for path in output_paths if cv2.imread(str(path)) is not None])
    summary_path = OUTPUT_DIR / "01_iou_summary.jpg"
    cv2.imwrite(str(summary_path), comparison)
    print(f"汇总图已保存：{summary_path}")

    cv2.imshow("IoU demo", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
