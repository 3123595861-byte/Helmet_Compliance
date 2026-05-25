from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "data" / "images" / "site.jpg"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "module2"


LOWER_YELLOW = np.array([20, 80, 80], dtype=np.uint8)
UPPER_YELLOW = np.array([35, 255, 255], dtype=np.uint8)
KERNEL_SIZE = 5


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
            "python src/module2_traditional_cv/03_morphology.py D:/your_image.jpg"
        )

    return image


def build_raw_yellow_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Build a raw yellow mask without morphology."""
    # cv2.cvtColor(src, code)
    # src: 输入 BGR 图像。
    # code=cv2.COLOR_BGR2HSV: 将 BGR 转换到 HSV 色彩空间。
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # cv2.inRange(src, lowerb, upperb)
    # lowerb: HSV 下界。
    # upperb: HSV 上界。
    # 输出 mask 中，符合范围的像素为 255，否则为 0。
    return cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)


def create_kernel(size: int = KERNEL_SIZE) -> np.ndarray:
    """Create an elliptical structuring element for morphology."""
    # cv2.getStructuringElement(shape, ksize)
    # shape=cv2.MORPH_ELLIPSE: 椭圆形结构元素，适合处理安全帽这类弧形目标。
    # ksize=(size, size): 核大小。核越大，修复越强，但也越容易让相邻物体粘连。
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def apply_morphology(mask: np.ndarray, kernel: np.ndarray) -> dict[str, np.ndarray]:
    """Apply erosion, dilation, opening, closing, and combined repair."""
    # cv2.erode(src, kernel, iterations)
    # 腐蚀会让白色区域变小，可去除细小白噪点，但也会削弱目标边缘。
    eroded = cv2.erode(mask, kernel, iterations=1)

    # cv2.dilate(src, kernel, iterations)
    # 膨胀会让白色区域变大，可连接断裂边缘，但也会放大噪声。
    dilated = cv2.dilate(mask, kernel, iterations=1)

    # cv2.morphologyEx(src, op, kernel)
    # op=cv2.MORPH_OPEN: 开运算，先腐蚀再膨胀，适合去掉小白点。
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # op=cv2.MORPH_CLOSE: 闭运算，先膨胀再腐蚀，适合填补安全帽 Mask 里的小黑洞。
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 常见组合：先开运算去噪，再闭运算补洞。
    opened_then_closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return {
        "raw": mask,
        "eroded": eroded,
        "dilated": dilated,
        "opened": opened,
        "closed": closed,
        "opened_then_closed": opened_then_closed,
    }


def draw_contours_on_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw contours found from a repaired mask on the original image."""
    result = image_bgr.copy()

    # cv2.findContours(image, mode, method)
    # image: 输入二值图。
    # mode=cv2.RETR_EXTERNAL: 只检测外层轮廓。
    # method=cv2.CHAIN_APPROX_SIMPLE: 压缩轮廓点。
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # cv2.drawContours(image, contours, contourIdx, color, thickness)
        # color=(255, 0, 0): BGR 蓝色，表示轮廓线。
        cv2.drawContours(result, [contour], -1, (255, 0, 0), 2)

        # cv2.rectangle(img, pt1, pt2, color, thickness)
        # color=(0, 255, 0): BGR 绿色，表示外接矩形。
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return result


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    """Add a title to an image for comparison display."""
    result = image.copy()

    if result.ndim == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    # cv2.putText(img, text, org, fontFace, fontScale, color, thickness)
    # org=(15, 35): 文字左下角坐标。
    # color=(0, 255, 0): BGR 绿色。
    cv2.putText(result, title, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return result


def resize_for_display(image: np.ndarray, max_width: int = 480) -> np.ndarray:
    """Resize image while keeping aspect ratio."""
    height, width = image.shape[:2]

    if width <= max_width:
        return image

    scale = max_width / width
    new_height = int(height * scale)

    # cv2.resize(src, dsize, interpolation)
    # dsize 的格式是 (width, height)。
    return cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)


def make_comparison_grid(images: dict[str, np.ndarray], contour_result: np.ndarray) -> np.ndarray:
    """Create a visual grid for morphology comparison."""
    raw = resize_for_display(add_title(images["raw"], "raw mask"))
    eroded = resize_for_display(add_title(images["eroded"], "erode"))
    dilated = resize_for_display(add_title(images["dilated"], "dilate"))
    opened = resize_for_display(add_title(images["opened"], "open"))
    closed = resize_for_display(add_title(images["closed"], "close"))
    repaired = resize_for_display(add_title(images["opened_then_closed"], "open + close"))
    contour = resize_for_display(add_title(contour_result, "contours after repair"))

    blank = np.zeros_like(raw)
    row1 = np.hstack([raw, eroded, dilated])
    row2 = np.hstack([opened, closed, repaired])
    row3 = np.hstack([contour, blank, blank])
    return np.vstack([row1, row2, row3])


def main() -> None:
    import sys

    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
    image = load_image(image_path)

    print(f"图片路径：{image_path}")
    print(f"图片矩阵 shape：{image.shape}")
    print("本关目标：观察腐蚀、膨胀、开运算、闭运算对黄色安全帽 Mask 的影响。")
    print(f"HSV 黄色范围：lower={LOWER_YELLOW.tolist()}, upper={UPPER_YELLOW.tolist()}")
    print(f"形态学核大小：{KERNEL_SIZE}x{KERNEL_SIZE}")

    raw_mask = build_raw_yellow_mask(image)
    kernel = create_kernel(KERNEL_SIZE)
    morph_images = apply_morphology(raw_mask, kernel)
    contour_result = draw_contours_on_mask(image, morph_images["opened_then_closed"])
    comparison_grid = make_comparison_grid(morph_images, contour_result)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "raw": OUTPUT_DIR / "03_raw_yellow_mask.jpg",
        "eroded": OUTPUT_DIR / "03_eroded_mask.jpg",
        "dilated": OUTPUT_DIR / "03_dilated_mask.jpg",
        "opened": OUTPUT_DIR / "03_opened_mask.jpg",
        "closed": OUTPUT_DIR / "03_closed_mask.jpg",
        "opened_then_closed": OUTPUT_DIR / "03_opened_then_closed_mask.jpg",
        "contour_result": OUTPUT_DIR / "03_contours_after_morphology.jpg",
        "comparison_grid": OUTPUT_DIR / "03_morphology_comparison_grid.jpg",
    }

    for key, path in output_paths.items():
        if key == "contour_result":
            cv2.imwrite(str(path), contour_result)
        elif key == "comparison_grid":
            cv2.imwrite(str(path), comparison_grid)
        else:
            cv2.imwrite(str(path), morph_images[key])
        print(f"{key} 已保存：{path}")

    cv2.imshow("Morphology comparison", comparison_grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
