from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "data" / "images" / "site.jpg"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "module1"


# OpenCV 的 HSV 范围：
# H: 0 到 179，表示色调。黄色通常大约在 20 到 35。
# S: 0 到 255，表示饱和度。数值越大，颜色越“纯”。
# V: 0 到 255，表示亮度。数值越大，画面越亮。
LOWER_YELLOW = np.array([20, 80, 80], dtype=np.uint8)
UPPER_YELLOW = np.array([35, 255, 255], dtype=np.uint8)


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
            "python src/module1_opencv_basics/02_hsv_mask.py D:/your_image.jpg"
        )

    return image


def build_yellow_mask(
    image_bgr: np.ndarray,
    lower_hsv: np.ndarray = LOWER_YELLOW,
    upper_hsv: np.ndarray = UPPER_YELLOW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert BGR image to HSV and generate a yellow binary mask."""
    # cv2.cvtColor(src, code)
    # src: 输入图像。
    # code=cv2.COLOR_BGR2HSV: 把 OpenCV 默认的 BGR 色彩空间转换为 HSV。
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # cv2.inRange(src, lowerb, upperb)
    # src: 输入图像，这里是 HSV 图。
    # lowerb: 每个通道允许的最小值，例如 [H_min, S_min, V_min]。
    # upperb: 每个通道允许的最大值，例如 [H_max, S_max, V_max]。
    # 返回值 mask: 二值图，范围内的像素是 255，范围外的像素是 0。
    raw_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # cv2.getStructuringElement(shape, ksize)
    # shape=cv2.MORPH_ELLIPSE: 椭圆形卷积核，适合修复安全帽这类圆弧物体。
    # ksize=(5, 5): 核大小是 5x5，越大修复越强，但也可能粘连不同物体。
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # cv2.morphologyEx(src, op, kernel)
    # op=cv2.MORPH_OPEN: 开运算，先腐蚀再膨胀，用来去掉小白噪点。
    opened_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)

    # op=cv2.MORPH_CLOSE: 闭运算，先膨胀再腐蚀，用来填补目标内部的小黑洞。
    cleaned_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)

    return hsv, raw_mask, cleaned_mask


def extract_masked_region(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep only pixels selected by the binary mask."""
    # cv2.bitwise_and(src1, src2, mask)
    # src1/src2: 要做按位与的两张图。这里传同一张原图，表示保留原图颜色。
    # mask: 单通道二值图。mask=255 的位置保留，mask=0 的位置变黑。
    return cv2.bitwise_and(image_bgr, image_bgr, mask=mask)


def draw_yellow_contours(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw bounding boxes around yellow regions for easier inspection."""
    result = image_bgr.copy()

    # cv2.findContours(image, mode, method)
    # image: 输入二值图，通常是 mask。
    # mode=cv2.RETR_EXTERNAL: 只找最外层轮廓，忽略内部嵌套轮廓。
    # method=cv2.CHAIN_APPROX_SIMPLE: 压缩轮廓点，减少内存占用。
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue

        # cv2.boundingRect(array)
        # array: 一个轮廓点集。
        # 返回 x, y, w, h，其中 x/y 是左上角坐标，w/h 是宽高。
        x, y, w, h = cv2.boundingRect(contour)

        # cv2.rectangle(img, pt1, pt2, color, thickness)
        # color=(0, 255, 0): BGR 绿色。
        # thickness=2: 矩形边框宽度是 2 像素。
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return result


def resize_for_display(image: np.ndarray, max_width: int = 480) -> np.ndarray:
    """Resize image for side-by-side display."""
    height, width = image.shape[:2]

    if width <= max_width:
        return image

    scale = max_width / width
    new_height = int(height * scale)

    # cv2.resize(src, dsize, interpolation)
    # dsize 的顺序是 (width, height)，不是 (height, width)。
    return cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)


def gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    """Convert a single-channel mask to BGR so it can be stacked with color images."""
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def main() -> None:
    import sys

    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
    image = load_image(image_path)

    print(f"图片路径：{image_path}")
    print(f"图片矩阵 shape：{image.shape}")
    print("本关目标：把 BGR 图像转换到 HSV，然后筛选黄色安全帽候选区域。")
    print(f"黄色 HSV 下界：{LOWER_YELLOW.tolist()}")
    print(f"黄色 HSV 上界：{UPPER_YELLOW.tolist()}")

    _, raw_mask, cleaned_mask = build_yellow_mask(image)
    yellow_region = extract_masked_region(image, cleaned_mask)
    boxed_result = draw_yellow_contours(image, cleaned_mask)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_mask_path = OUTPUT_DIR / "02_yellow_raw_mask.jpg"
    cleaned_mask_path = OUTPUT_DIR / "02_yellow_cleaned_mask.jpg"
    region_path = OUTPUT_DIR / "02_yellow_region.jpg"
    boxed_path = OUTPUT_DIR / "02_yellow_boxed_result.jpg"

    cv2.imwrite(str(raw_mask_path), raw_mask)
    cv2.imwrite(str(cleaned_mask_path), cleaned_mask)
    cv2.imwrite(str(region_path), yellow_region)
    cv2.imwrite(str(boxed_path), boxed_result)

    print(f"原始 Mask 已保存：{raw_mask_path}")
    print(f"清理后 Mask 已保存：{cleaned_mask_path}")
    print(f"黄色区域提取结果已保存：{region_path}")
    print(f"黄色候选框结果已保存：{boxed_path}")

    row1 = np.hstack(
        [
            resize_for_display(image),
            resize_for_display(gray_to_bgr(raw_mask)),
        ]
    )
    row2 = np.hstack(
        [
            resize_for_display(yellow_region),
            resize_for_display(boxed_result),
        ]
    )

    display = np.vstack([row1, row2])

    # 窗口布局：
    # 左上：原图；右上：原始黄色 Mask。
    # 左下：只保留黄色区域；右下：用绿框标出黄色候选区域。
    cv2.imshow("HSV yellow mask: original | mask | yellow region | boxes", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
