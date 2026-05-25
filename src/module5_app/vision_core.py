from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# 常量配置
# ==========================================
PERSON_CLASS_ID = 0
LOWER_YELLOW = np.array([20, 80, 80], dtype=np.uint8)
UPPER_YELLOW = np.array([35, 255, 255], dtype=np.uint8)

# ==========================================
# 数据结构
# ==========================================
@dataclass
class PersonDetection:
    box: tuple[int, int, int, int]
    confidence: float

@dataclass
class ComplianceResult:
    person_id: int
    person_box: tuple[int, int, int, int]
    head_box: tuple[int, int, int, int]
    helmet_box: tuple[int, int, int, int] | None
    yellow_ratio: float
    compliant: bool
    violation_type: str
    confidence: float

# ==========================================
# 核心视觉算法
# ==========================================
def detect_persons(model: YOLO, frame: np.ndarray, conf: float) -> list[PersonDetection]:
    """使用 YOLO 检测画面中的人员"""
    results = model(frame, conf=conf, verbose=False)
    if not results or results[0].boxes is None:
        return []
    persons: list[PersonDetection] = []
    for box, score, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
        if int(cls.item()) == PERSON_CLASS_ID:
            x1, y1, x2, y2 = map(int, box.tolist())
            persons.append(PersonDetection((x1, y1, x2, y2), float(score.item())))
    return persons

def estimate_head(person_box: tuple[int, int, int, int], shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """根据人体框估算头部区域"""
    x1, y1, x2, y2 = person_box
    h, w = shape[:2]
    pw, ph = max(1, x2 - x1), max(1, y2 - y1)
    cx = (x1 + x2) // 2
    hw, hh = int(pw * 0.62), int(ph * 0.28)
    return max(0, cx - hw // 2), max(0, y1), min(w, cx + hw // 2), min(h, y1 + hh)

def yellow_check(frame: np.ndarray, box: tuple[int, int, int, int], threshold: float) -> tuple[bool, float, tuple[int, int, int, int] | None]:
    """检测指定区域内的黄色像素占比"""
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False, 0.0, None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    ratio = cv2.countNonZero(mask) / max(1, mask.shape[0] * mask.shape[1])
    helmet_box = None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 20:
            bx, by, bw, bh = cv2.boundingRect(c)
            helmet_box = (x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh)
    return ratio >= threshold, ratio, helmet_box

def evaluate(frame: np.ndarray, persons: list[PersonDetection], yellow_th: float) -> list[ComplianceResult]:
    """综合评估画面中所有人员的合规状态"""
    results: list[ComplianceResult] = []
    for pid, person in enumerate(persons, 1):
        hb = estimate_head(person.box, frame.shape)
        ok, ratio, helmet = yellow_check(frame, hb, yellow_th)
        violation = "" if ok else "no_yellow_helmet_in_head_region"
        confidence = ratio if ok else max(0.0, 1.0 - ratio / max(yellow_th, 1e-6))
        results.append(ComplianceResult(pid, person.box, hb, helmet, ratio, ok, violation, confidence))
    return results

def draw(frame: np.ndarray, persons: list[PersonDetection], results: list[ComplianceResult], fps: float, alarm: bool) -> np.ndarray:
    """在画面上绘制检测框和数据结果"""
    vis = frame.copy()
    by_id = {r.person_id: r for r in results}
    for pid, person in enumerate(persons, 1):
        r = by_id.get(pid)
        color = (0, 200, 0) if r and r.compliant else (0, 0, 255)
        x1, y1, x2, y2 = person.box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"person {person.confidence:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if r:
            hx1, hy1, hx2, hy2 = r.head_box
            cv2.rectangle(vis, (hx1, hy1), (hx2, hy2), (255, 0, 0), 2)
            cv2.putText(vis, f"yellow={r.yellow_ratio:.3f}", (hx1, hy2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            if r.helmet_box:
                bx1, by1, bx2, by2 = r.helmet_box
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
    cv2.putText(vis, "ALARM ACTIVE" if alarm else "NORMAL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) if alarm else (0, 180, 0), 2)
    cv2.putText(vis, f"FPS: {fps:.1f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)
    return vis