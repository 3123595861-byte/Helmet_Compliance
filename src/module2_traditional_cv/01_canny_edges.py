from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "data" / "images" / "site.jpg"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "module2"


CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150


def load_image(image_path: Path) -> np.ndarray:
    """Read an image with OpenCV and fail clearly if the path is wrong."""
    # cv2.imread(filename, flags)
    # filename: 图片路径字符串。
    # flags=cv2.IMREAD_COLOR: 以 BGR 三通道彩色图读取。
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(
            f"没有读到图片：{image_path}\n"
            "请先创建 data/images 文件夹，并放入一张名为 site.jpg 的工地图片；\n"
            "或者运行脚本时传入图片路径，例如：\n"
            "python src/module2_traditional_cv/01_canny_edges.py D:/your_image.jpg"
        )

    return image


def preprocess_for_edges(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert BGR image to grayscale and blur it before edge detection."""
    # cv2.cvtColor(src, code)
    # src: 输入图像。
    # code=cv2.COLOR_BGR2GRAY: 将 BGR 三通道彩色图转换为单通道灰度图。
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # cv2.GaussianBlur(src, ksize, sigmaX)
    # src: 输入图像。
    # ksize=(5, 5): 高斯核大小，必须是正奇数。核越大，平滑越强。
    # sigmaX=1.4: X 方向标准差，控制模糊程度。
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    return gray, blurred


def compute_sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude with Sobel operators for teaching inspection."""
    # cv2.Sobel(src, ddepth, dx, dy, ksize)
    # src: 输入灰度图。
    # ddepth=cv2.CV_64F: 输出使用 64 位浮点数，避免梯度为负时被截断。
    # dx=1, dy=0: 计算 x 方向梯度，也就是左右方向灰度变化。
    # ksize=3: Sobel 卷积核大小。
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

    # dx=0, dy=1: 计算 y 方向梯度，也就是上下方向灰度变化。
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # np.sqrt(grad_x ** 2 + grad_y ** 2) 对应梯度模长：
    # 梯度越大，说明像素灰度变化越剧烈，越可能是边缘。
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # cv2.normalize(src, dst, alpha, beta, norm_type)
    # alpha=0, beta=255: 把数值归一化到 0 到 255，方便显示和保存。
    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    return normalized.astype(np.uint8)


def run_canny(gray_or_blurred: np.ndarray, low_threshold: int, high_threshold: int) -> np.ndarray:
    """Run Canny edge detection with given thresholds."""
    # cv2.Canny(image, threshold1, threshold2)
    # image: 输入单通道灰度图，通常先做高斯模糊以减少噪声。
    # threshold1: 低阈值。弱边缘低于该值会被丢弃。
    # threshold2: 高阈值。强边缘高于该值会被保留。
    # 介于二者之间的边缘，如果和强边缘连通，也会被保留。
    return cv2.Canny(gray_or_blurred, low_threshold, high_threshold)


def make_threshold_comparison(blurred: np.ndarray) -> np.ndarray:
    """Generate a comparison image with multiple Canny threshold pairs."""
    threshold_pairs = [
        (30, 90),
        (50, 150),
        (80, 200),
        (120, 240),
    ]

    edge_images = []
    for low, high in threshold_pairs:
        edges = run_canny(blurred, low, high)
        edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # cv2.putText(img, text, org, fontFace, fontScale, color, thickness)
        # org=(20, 40): 文字左下角坐标。
        # fontScale=1.0: 字体缩放比例。
        # color=(0, 255, 0): BGR 绿色。
        cv2.putText(
            edge_bgr,
            f"Canny {low}, {high}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        edge_images.append(edge_bgr)

    top_row = np.hstack([edge_images[0], edge_images[1]])
    bottom_row = np.hstack([edge_images[2], edge_images[3]])
    return np.vstack([top_row, bottom_row])


def overlay_edges_on_image(image_bgr: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Overlay detected edges on the original image in red."""
    overlay = image_bgr.copy()

    # edges > 0 会得到一个布尔矩阵，True 表示该像素是边缘。
    # OpenCV 使用 BGR，因此 (0, 0, 255) 是红色。
    overlay[edges > 0] = (0, 0, 255)

    return overlay


def resize_for_display(image: np.ndarray, max_width: int = 600) -> np.ndarray:
    """Resize image while keeping aspect ratio."""
    height, width = image.shape[:2]

    if width <= max_width:
        return image

    scale = max_width / width
    new_height = int(height * scale)

    # cv2.resize(src, dsize, interpolation)
    # dsize: 输出尺寸，格式是 (width, height)。
    # interpolation=cv2.INTER_AREA: 缩小图片时常用，视觉效果稳定。
    return cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)


def gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    """Convert a single-channel gray image to BGR for stacking display."""
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def main() -> None:
    import sys

    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
    image = load_image(image_path)

    print(f"图片路径：{image_path}")
    print(f"图片矩阵 shape：{image.shape}")
    print("本关目标：理解灰度图、Sobel 梯度和 Canny 边缘检测。")
    print(f"默认 Canny 低阈值：{CANNY_LOW_THRESHOLD}")
    print(f"默认 Canny 高阈值：{CANNY_HIGH_THRESHOLD}")

    gray, blurred = preprocess_for_edges(image)
    sobel_magnitude = compute_sobel_magnitude(blurred)
    edges = run_canny(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    edge_overlay = overlay_edges_on_image(image, edges)
    threshold_comparison = make_threshold_comparison(blurred)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gray_path = OUTPUT_DIR / "01_gray.jpg"
    blurred_path = OUTPUT_DIR / "01_blurred.jpg"
    sobel_path = OUTPUT_DIR / "01_sobel_magnitude.jpg"
    edges_path = OUTPUT_DIR / "01_canny_edges.jpg"
    overlay_path = OUTPUT_DIR / "01_canny_overlay.jpg"
    comparison_path = OUTPUT_DIR / "01_canny_threshold_comparison.jpg"

    cv2.imwrite(str(gray_path), gray)
    cv2.imwrite(str(blurred_path), blurred)
    cv2.imwrite(str(sobel_path), sobel_magnitude)
    cv2.imwrite(str(edges_path), edges)
    cv2.imwrite(str(overlay_path), edge_overlay)
    cv2.imwrite(str(comparison_path), threshold_comparison)

    print(f"灰度图已保存：{gray_path}")
    print(f"高斯模糊图已保存：{blurred_path}")
    print(f"Sobel 梯度图已保存：{sobel_path}")
    print(f"Canny 边缘图已保存：{edges_path}")
    print(f"边缘叠加图已保存：{overlay_path}")
    print(f"多阈值对比图已保存：{comparison_path}")

    row1 = np.hstack(
        [
            resize_for_display(image),
            resize_for_display(gray_to_bgr(gray)),
        ]
    )
    row2 = np.hstack(
        [
            resize_for_display(gray_to_bgr(sobel_magnitude)),
            resize_for_display(gray_to_bgr(edges)),
        ]
    )
    display = np.vstack([row1, row2])

    # 窗口布局：
    # 左上：原图；右上：灰度图。
    # 左下：Sobel 梯度强度；右下：Canny 边缘。
    cv2.imshow("Canny basics: original | gray | sobel | canny", display)
    cv2.imshow("Canny threshold comparison", resize_for_display(threshold_comparison, 1000))
    cv2.imshow("Edges overlay on original", resize_for_display(edge_overlay, 1000))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
