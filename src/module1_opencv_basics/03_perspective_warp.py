from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "data" / "images" / "site.jpg"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "module1"


WINDOW_NAME = "click 4 source points: top-left, top-right, bottom-right, bottom-left"
MAX_DISPLAY_WIDTH = 1100


class PointSelector:
    """Collect four mouse-clicked points from an image window."""

    def __init__(self, original_image: np.ndarray, display_image: np.ndarray, scale: float) -> None:
        self.original_image = original_image
        self.display_image = display_image.copy()
        self.preview_image = display_image.copy()
        self.scale = scale
        self.points_on_display: list[tuple[int, int]] = []

    def on_mouse(self, event: int, x: int, y: int, flags: int, userdata: object) -> None:
        # cv2.setMouseCallback 会把鼠标事件传给这个函数。
        # event: 鼠标事件类型，例如 cv2.EVENT_LBUTTONDOWN 表示左键按下。
        # x/y: 鼠标在显示窗口中的坐标。
        # flags: 鼠标拖拽、组合键等状态，本例暂时不用。
        # userdata: 用户自定义数据，本例暂时不用。
        del flags, userdata

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(self.points_on_display) >= 4:
            return

        self.points_on_display.append((x, y))
        self._redraw_preview()

    def _redraw_preview(self) -> None:
        self.preview_image = self.display_image.copy()

        for index, point in enumerate(self.points_on_display, start=1):
            # cv2.circle(img, center, radius, color, thickness)
            # center: 圆心坐标，格式是 (x, y)。
            # radius: 半径，单位是像素。
            # color=(0, 0, 255): BGR 红色。
            # thickness=-1: 填充整个圆。
            cv2.circle(self.preview_image, point, 6, (0, 0, 255), -1)

            # cv2.putText(img, text, org, fontFace, fontScale, color, thickness)
            # org: 文字左下角坐标。
            # fontScale: 字体缩放比例。
            cv2.putText(
                self.preview_image,
                str(index),
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        if len(self.points_on_display) >= 2:
            # cv2.polylines(img, pts, isClosed, color, thickness)
            # pts: 多边形点集，需要是 int32 数组。
            # isClosed: 是否闭合。点数小于 4 时不闭合，方便观察点击顺序。
            # color=(0, 255, 0): BGR 绿色。
            pts = np.array(self.points_on_display, dtype=np.int32)
            cv2.polylines(
                self.preview_image,
                [pts],
                len(self.points_on_display) == 4,
                (0, 255, 0),
                2,
            )

    def get_points_on_original(self) -> np.ndarray:
        """Convert clicked display coordinates back to original image coordinates."""
        if len(self.points_on_display) != 4:
            raise ValueError("需要点击 4 个点才能做透视变换。")

        points = np.array(self.points_on_display, dtype=np.float32)
        return points / self.scale


def load_image(image_path: Path) -> np.ndarray:
    """Read an image with OpenCV and fail clearly if the path is wrong."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(
            f"没有读到图片：{image_path}\n"
            "请先创建 data/images 文件夹，并放入一张名为 site.jpg 的工地图片；\n"
            "或者运行脚本时传入图片路径，例如：\n"
            "python src/module1_opencv_basics/03_perspective_warp.py D:/your_image.jpg"
        )

    return image


def resize_for_selection(image: np.ndarray, max_width: int = MAX_DISPLAY_WIDTH) -> tuple[np.ndarray, float]:
    """Resize image for point selection and return the scale ratio."""
    height, width = image.shape[:2]

    if width <= max_width:
        return image.copy(), 1.0

    scale = max_width / width
    new_height = int(height * scale)

    resized = cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def order_points(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)

    coordinate_sum = points.sum(axis=1)
    coordinate_diff = np.diff(points, axis=1).reshape(-1)

    rect[0] = points[np.argmin(coordinate_sum)]
    rect[2] = points[np.argmax(coordinate_sum)]
    rect[1] = points[np.argmin(coordinate_diff)]
    rect[3] = points[np.argmax(coordinate_diff)]

    return rect


def compute_output_size(ordered_points: np.ndarray) -> tuple[int, int]:
    """Estimate output width and height from the selected quadrilateral."""
    top_left, top_right, bottom_right, bottom_left = ordered_points

    top_width = np.linalg.norm(top_right - top_left)
    bottom_width = np.linalg.norm(bottom_right - bottom_left)
    output_width = int(max(top_width, bottom_width))

    left_height = np.linalg.norm(bottom_left - top_left)
    right_height = np.linalg.norm(bottom_right - top_right)
    output_height = int(max(left_height, right_height))

    if output_width <= 0 or output_height <= 0:
        raise ValueError("输出尺寸不合法，请检查四个点是否点击正确。")

    return output_width, output_height


def perspective_warp(image: np.ndarray, source_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warp a quadrilateral region to a front-view rectangle."""
    ordered_source = order_points(source_points)
    output_width, output_height = compute_output_size(ordered_source)

    destination_points = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )

    # cv2.getPerspectiveTransform(src, dst)
    # src: 原图中的四个点，类型通常是 float32，顺序应为左上、右上、右下、左下。
    # dst: 目标图中的四个点，表示要把 src 映射到哪个矩形位置。
    # 返回值 matrix: 3x3 透视变换矩阵。
    matrix = cv2.getPerspectiveTransform(ordered_source, destination_points)

    # cv2.warpPerspective(src, M, dsize)
    # src: 输入图像。
    # M: 3x3 透视变换矩阵。
    # dsize: 输出图像尺寸，格式是 (width, height)。
    warped = cv2.warpPerspective(image, matrix, (output_width, output_height))

    return warped, ordered_source, destination_points


def select_four_points(image: np.ndarray) -> np.ndarray:
    """Let user click four points on a resized image."""
    display_image, scale = resize_for_selection(image)
    selector = PointSelector(image, display_image, scale)

    # cv2.namedWindow(winname, flags)
    # flags=cv2.WINDOW_NORMAL: 允许窗口缩放。
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # cv2.setMouseCallback(winname, onMouse)
    # winname: 要绑定鼠标事件的窗口名。
    # onMouse: 鼠标回调函数。
    cv2.setMouseCallback(WINDOW_NAME, selector.on_mouse)

    print("请按顺序点击 4 个点：左上 -> 右上 -> 右下 -> 左下。")
    print("如果点错了，按 r 重置；点击满 4 个点后按 Enter 确认；按 Esc 退出。")

    while True:
        cv2.imshow(WINDOW_NAME, selector.preview_image)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            raise KeyboardInterrupt("用户取消了透视变换。")

        if key in (ord("r"), ord("R")):
            selector.points_on_display.clear()
            selector._redraw_preview()
            print("已重置点位，请重新点击 4 个点。")

        if key in (13, 10) and len(selector.points_on_display) == 4:
            break

    cv2.destroyWindow(WINDOW_NAME)
    return selector.get_points_on_original()


def draw_selected_polygon(image: np.ndarray, ordered_points: np.ndarray) -> np.ndarray:
    """Draw selected quadrilateral on original image."""
    result = image.copy()
    pts = ordered_points.astype(np.int32)

    cv2.polylines(result, [pts], True, (0, 255, 0), 3)

    labels = ["TL", "TR", "BR", "BL"]
    for label, point in zip(labels, pts):
        x, y = point
        cv2.circle(result, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(result, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return result


def main() -> None:
    import sys

    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
    image = load_image(image_path)

    print(f"图片路径：{image_path}")
    print(f"图片矩阵 shape：{image.shape}")
    print("本关目标：选择原图中的一个倾斜四边形区域，把它校正为正视角矩形。")

    source_points = select_four_points(image)
    warped, ordered_source, destination_points = perspective_warp(image, source_points)
    selected_preview = draw_selected_polygon(image, ordered_source)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_path = OUTPUT_DIR / "03_selected_perspective_points.jpg"
    warped_path = OUTPUT_DIR / "03_perspective_warp_result.jpg"

    cv2.imwrite(str(selected_path), selected_preview)
    cv2.imwrite(str(warped_path), warped)

    print("原图四点，顺序为左上、右上、右下、左下：")
    print(ordered_source)
    print("目标图四点：")
    print(destination_points)
    print(f"四点标注图已保存：{selected_path}")
    print(f"透视校正结果已保存：{warped_path}")

    preview_resized, _ = resize_for_selection(selected_preview, 600)
    warped_resized, _ = resize_for_selection(warped, 600)

    min_height = min(preview_resized.shape[0], warped_resized.shape[0])
    preview_resized = preview_resized[:min_height, :]
    warped_resized = warped_resized[:min_height, :]
    display = np.hstack([preview_resized, warped_resized])

    cv2.imshow("left: selected quadrilateral | right: perspective warped", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
