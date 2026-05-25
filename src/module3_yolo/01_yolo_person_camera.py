from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from time import time
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "yolov8n.pt"
CONF_THRESHOLD = 0.25
CAMERA_INDEX = 0
WINDOW_NAME = "YOLO person detection"
TARGET_CLASS_NAME = "person"
TARGET_CLASS_ID = 0
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "module3"


@dataclass
class DetectionResult:
    """Store one detection after post-processing."""
    box: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


def load_model() -> YOLO:
    """Load the official pre-trained YOLO model."""
    # YOLO(model_path)
    # model_path: 权重文件路径或官方模型名称，例如 yolov8n.pt。
    return YOLO(MODEL_NAME)


def open_camera(camera_index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    """Open webcam capture and fail clearly if it cannot be opened."""
    # cv2.VideoCapture(index)
    # index=0: 通常表示默认摄像头。
    # 如果你有多个摄像头，可以尝试 1、2。
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开摄像头 {camera_index}。\n"
            "请检查摄像头是否被占用，或者把 CAMERA_INDEX 改成 1、2 试试。"
        )
    return cap


def load_image(image_path: Path) -> np.ndarray:
    """Read one image from disk for YOLO inference."""
    # cv2.imread(filename, flags)
    # filename: 图片路径字符串。
    # flags=cv2.IMREAD_COLOR: 按 BGR 三通道彩色图读取。
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"没有读到图片：{image_path}")
    return image


def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Prepare a BGR frame for YOLO inference."""
    # Ultralytics 的 YOLO 可以直接接收 OpenCV 读出的 BGR 图像。
    # 内部会自动完成 resize、归一化、转 tensor 等预处理。
    return frame_bgr

    
def run_inference(model: YOLO, frame_bgr: np.ndarray):
    """Run YOLO inference on one frame."""
    # model(source, conf, verbose)
    # source: 输入图片，可以是 numpy 图像矩阵，也可以是图片路径。
    # conf: 置信度阈值，低于该值的检测框会被过滤。
    # verbose=False: 关闭模型内部打印，避免终端刷屏。
    return model(frame_bgr, conf=CONF_THRESHOLD, verbose=False)


def postprocess_results(results, target_class_id: int = TARGET_CLASS_ID) -> list[DetectionResult]:
    """Extract target-class detections from raw YOLO results."""
    detections: list[DetectionResult] = []
    if not results:
        return detections
    result = results[0]
    names = result.names
    if result.boxes is None:
        return detections
    # result.boxes.xyxy: 检测框坐标，格式是 [x1, y1, x2, y2]。
    # result.boxes.conf: 每个检测框的置信度。
    # result.boxes.cls: 每个检测框的类别 id。
    for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
        class_id = int(cls.item())
        if class_id != target_class_id:
            continue
        x1, y1, x2, y2 = box.tolist()
        detections.append(
            DetectionResult(
                box=(int(x1), int(y1), int(x2), int(y2)),
                confidence=float(conf.item()),
                class_id=class_id,
                class_name=names[class_id],
            )
        )
    return detections


def draw_detections(frame_bgr: np.ndarray, detections: list[DetectionResult], fps: float | None = None) -> np.ndarray:
    """Draw detections on the frame or image."""
    result = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.box
        # cv2.rectangle(img, pt1, pt2, color, thickness)
        # pt1/pt2: 左上角和右下角坐标，格式都是 (x, y)。
        # color=(0, 255, 0): OpenCV 使用 BGR 顺序，这里表示绿色。
        # thickness=2: 边框粗细 2 像素。
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        # cv2.putText(img, text, org, fontFace, fontScale, color, thickness)
        # org: 文字左下角坐标。
        # fontScale: 字体缩放比例。
        cv2.putText(
            result,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    if fps is not None:
        cv2.putText(
            result,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
    return result


def compute_fps(prev_time: float) -> tuple[float, float]:
    """Compute FPS from current time."""
    now = time()
    delta = max(now - prev_time, 1e-6)
    fps = 1.0 / delta
    return fps, now


def detect_image(model: YOLO, image_path: Path) -> None:
    """Detect persons in one existing image and save the visualized result."""
    image = load_image(image_path)
    preprocessed = preprocess_frame(image)
    results = run_inference(model, preprocessed)
    detections = postprocess_results(results, TARGET_CLASS_ID)
    vis = draw_detections(image, detections)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"01_yolo_person_{image_path.stem}.jpg"
    # cv2.imwrite(filename, img)
    # filename: 保存路径。
    # img: 要保存的 BGR 图像矩阵。
    cv2.imwrite(str(output_path), vis)
    print(f"输入图片：{image_path}")
    print(f"检测到 person 数量：{len(detections)}")
    for index, det in enumerate(detections, start=1):
        print(f"person {index}: box={det.box}, confidence={det.confidence:.3f}")
    print(f"结果图已保存：{output_path}")
    cv2.imshow(WINDOW_NAME, vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_camera(model: YOLO) -> None:
    """Detect persons from webcam stream."""
    cap = open_camera(CAMERA_INDEX)
    print("YOLO 摄像头推理管道开始运行。")
    print("三个阶段分别是：Pre-processing -> Inference -> Post-processing。")
    print(f"当前目标类别：{TARGET_CLASS_NAME} (class id = {TARGET_CLASS_ID})")
    print("按 q 退出。")
    prev_time = time()
    try:
        while True:
            # cap.read() 返回 (ret, frame)
            # ret: 是否成功读取到一帧。
            # frame: OpenCV 读取到的 BGR 图像。
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头帧，程序结束。")
                break
            preprocessed = preprocess_frame(frame)
            results = run_inference(model, preprocessed)
            detections = postprocess_results(results, TARGET_CLASS_ID)
            fps, prev_time = compute_fps(prev_time)
            vis = draw_detections(frame, detections, fps)
            cv2.imshow(WINDOW_NAME, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    import sys
    model = load_model()
    if len(sys.argv) >= 2:
        detect_image(model, Path(sys.argv[1]))
    else:
        detect_camera(model)


if __name__ == "__main__":
    main()
