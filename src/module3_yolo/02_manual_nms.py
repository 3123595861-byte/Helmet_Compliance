from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "module3"


@dataclass
class BoxPrediction:
    """Store one predicted bounding box for NMS demonstration."""

    box: tuple[float, float, float, float]
    score: float
    label: str = "helmet"


@dataclass
class KeptBox:
    """Store one box that survives NMS."""

    box: tuple[float, float, float, float]
    score: float
    label: str


def compute_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Compute IoU between two boxes in xyxy format."""
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


def non_max_suppression(
    predictions: list[BoxPrediction],
    iou_threshold: float = 0.5,
) -> tuple[list[KeptBox], list[str]]:
    """Perform a simple class-agnostic NMS for teaching."""
    # 1. 按置信度从高到低排序。
    sorted_predictions = sorted(predictions, key=lambda pred: pred.score, reverse=True)
    kept: list[KeptBox] = []
    log_lines: list[str] = []

    while sorted_predictions:
        best = sorted_predictions.pop(0)
        kept.append(KeptBox(box=best.box, score=best.score, label=best.label))
        log_lines.append(f"保留框：{best.label}, score={best.score:.2f}, box={best.box}")

        remaining: list[BoxPrediction] = []
        for pred in sorted_predictions:
            iou = compute_iou(best.box, pred.box)
            if iou >= iou_threshold:
                log_lines.append(
                    f"删除框：{pred.label}, score={pred.score:.2f}, box={pred.box}, IoU={iou:.2f}"
                )
            else:
                remaining.append(pred)

        sorted_predictions = remaining

    return kept, log_lines


def draw_boxes(
    image: np.ndarray,
    boxes: list[KeptBox],
    color: tuple[int, int, int],
    title: str,
) -> np.ndarray:
    """Draw boxes and scores on a canvas."""
    result = image.copy()

    cv2.putText(
        result,
        title,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
    )

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.box)

        # cv2.rectangle(img, pt1, pt2, color, thickness)
        # pt1/pt2: 左上角和右下角坐标，格式是 (x, y)。
        # color: BGR 颜色。
        # thickness=2: 线条粗细为 2 像素。
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        label = f"{box.label} {box.score:.2f}"
        cv2.putText(
            result,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return result


def make_demo_canvas(width: int = 900, height: int = 600) -> np.ndarray:
    """Create a blank canvas and a few overlapping demo boxes."""
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)

    # 为了让 NMS 演示直观，这里手动构造三个重叠框。
    # 这三个框都很像同一个目标，但只有分数最高的那个应该被保留。
    return canvas


def get_demo_predictions() -> list[BoxPrediction]:
    """Create three overlapping boxes for the NMS demo."""
    return [
        BoxPrediction(box=(120, 120, 360, 360), score=0.92, label="helmet"),
        BoxPrediction(box=(150, 150, 380, 370), score=0.84, label="helmet"),
        BoxPrediction(box=(300, 130, 520, 350), score=0.77, label="helmet"),
    ]


def draw_demo_boxes(image: np.ndarray, predictions: list[BoxPrediction], color: tuple[int, int, int], title: str) -> np.ndarray:
    """Draw raw predicted boxes before NMS."""
    result = image.copy()
    cv2.putText(result, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    for pred in predictions:
        x1, y1, x2, y2 = map(int, pred.box)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            result,
            f"{pred.label} {pred.score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return result


def main() -> None:
    predictions = get_demo_predictions()
    kept, log_lines = non_max_suppression(predictions, iou_threshold=0.5)

    print("手写 NMS 演示开始")
    print("排序规则：先保留置信度最高的框，再删除与它 IoU 过高的重叠框。")
    print("IoU 阈值：0.5")
    print()

    for line in log_lines:
        print(line)

    print()
    print(f"最终保留框数量：{len(kept)}")
    for index, box in enumerate(kept, start=1):
        print(f"保留 {index}: score={box.score:.2f}, box={box.box}")

    canvas = make_demo_canvas()
    before = draw_demo_boxes(canvas, predictions, (0, 0, 255), "Before NMS")
    after = draw_boxes(canvas, kept, (0, 200, 0), "After NMS")

    comparison = np.hstack([before, after])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "02_manual_nms_demo.jpg"
    cv2.imwrite(str(output_path), comparison)

    print(f"对比图已保存：{output_path}")

    cv2.imshow("Manual NMS demo: before | after", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
