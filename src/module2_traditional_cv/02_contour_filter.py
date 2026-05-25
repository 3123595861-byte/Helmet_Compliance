from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "data" / "images" / "site.jpg"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "module2"


LOWER_YELLOW = np.array([20, 80, 80], dtype=np.uint8)
UPPER_YELLOW = np.array([35, 255, 255], dtype=np.uint8)
MIN_CONTOUR_AREA = 150.0
MAX_CONTOUR_AREA = 50000.0
MIN_CIRCULARITY = 0.35


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
            "python src/module2_traditional_cv/02_contour_filter.py D:/your_image.jpg"
        )

    return image


def build_yellow_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Build a cleaned yellow mask in HSV color space."""
    # cv2.cvtColor(src, code)
    # code=cv2.COLOR_BGR2HSV: 把 BGR 转换到 HSV，便于按颜色过滤。
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # cv2.inRange(src, lowerb, upperb)
    # lowerb/upperb: HSV 三个通道的下界和上界。
    # 返回二值图：范围内像素为 255，范围外像素为 0。
    mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)

    # 椭圆核更适合处理安全帽这类弧形目标。
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 开运算去除小白点噪声。
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 闭运算填补目标内部小黑洞。
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def calculate_circularity(area: float, perimeter: float) -> float:
    """Calculate circularity: 1.0 is close to a perfect circle."""
    if perimeter <= 0:
        return 0.0

    # 圆度公式：4 * pi * area / perimeter^2。
    # 完美圆形接近 1，细长形状接近 0。
    return float(4.0 * np.pi * area / (perimeter * perimeter))


def find_candidate_contours(mask: np.ndarray) -> list[dict[str, object]]:
    """Find contours and keep candidates by area and circularity."""
    # cv2.findContours(image, mode, method)
    # image: 输入二值图，通常白色区域是目标，黑色区域是背景。
    # mode=cv2.RETR_EXTERNAL: 只找最外层轮廓，减少内部洞的干扰。
    # method=cv2.CHAIN_APPROX_SIMPLE: 压缩水平、垂直、斜线上的冗余点。
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[dict[str, object]] = []

    for index, contour in enumerate(contours):
        # cv2.contourArea(contour)
        # contour: 单个轮廓点集。
        # 返回轮廓围成区域的面积，单位是像素平方。
        area = cv2.contourArea(contour)

        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue

        # cv2.arcLength(curve, closed)
        # curve: 轮廓点集。
        # closed=True: 表示轮廓是闭合曲线，计算闭合周长。
        perimeter = cv2.arcLength(contour, True)
        circularity = calculate_circularity(area, perimeter)

        if circularity < MIN_CIRCULARITY:
            continue

        # cv2.boundingRect(array)
        # 返回 x, y, w, h，分别是外接矩形左上角坐标、宽、高。
        x, y, w, h = cv2.boundingRect(contour)

        candidates.append(
            {
                "index": index,
                "contour": contour,
                "area": area,
                "perimeter": perimeter,
                "circularity": circularity,
                "box": (x, y, w, h),
            }
        )

    if hierarchy is not None:
        print(f"轮廓层次结构 shape：{hierarchy.shape}")
        print("使用 cv2.RETR_EXTERNAL 时，通常只保留最外层轮廓。")

    return candidates


def draw_candidates(image_bgr: np.ndarray, candidates: list[dict[str, object]]) -> np.ndarray:
    """Draw contour candidates and their metrics."""
    result = image_bgr.copy()

    for candidate in candidates:
        x, y, w, h = candidate["box"]
        area = float(candidate["area"])
        circularity = float(candidate["circularity"])
        contour = candidate["contour"]

        # cv2.drawContours(image, contours, contourIdx, color, thickness)
        # contours: 轮廓列表。
        # contourIdx=-1: 绘制列表中的全部轮廓。
        # color=(255, 0, 0): BGR 蓝色。
        # thickness=2: 线宽 2 像素。
        cv2.drawContours(result, [contour], -1, (255, 0, 0), 2)

        # cv2.rectangle(img, pt1, pt2, color, thickness)
        # pt1=(x, y): 左上角。
        # pt2=(x + w, y + h): 右下角。
        # color=(0, 255, 0): BGR 绿色。
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        label = f"A={area:.0f}, C={circularity:.2f}"
        cv2.putText(
            result,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    return result


def create_all_contours_debug(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw all raw contours before filtering for comparison."""
    debug = image_bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(image, contours, contourIdx, color, thickness)
    # contourIdx=-1 表示画出所有轮廓。
    cv2.drawContours(debug, contours, -1, (0, 0, 255), 1)

    return debug


def resize_for_display(image: np.ndarray, max_width: int = 600) -> np.ndarray:
    """Resize image while keeping aspect ratio."""
    height, width = image.shape[:2]

    if width <= max_width:
        return image

    scale = max_width / width
    new_height = int(height * scale)

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
    print("本关目标：根据面积、周长、圆度，从黄色 Mask 中筛选近似圆形目标。")
    print(f"面积范围：{MIN_CONTOUR_AREA} 到 {MAX_CONTOUR_AREA}")
    print(f"最小圆度：{MIN_CIRCULARITY}")

    mask = build_yellow_mask(image)
    candidates = find_candidate_contours(mask)
    all_contours_debug = create_all_contours_debug(image, mask)
    candidate_result = draw_candidates(image, candidates)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mask_path = OUTPUT_DIR / "02_yellow_mask_for_contours.jpg"
    all_contours_path = OUTPUT_DIR / "02_all_contours_debug.jpg"
    candidates_path = OUTPUT_DIR / "02_circular_candidates.jpg"

    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(all_contours_path), all_contours_debug)
    cv2.imwrite(str(candidates_path), candidate_result)

    print(f"黄色 Mask 已保存：{mask_path}")
    print(f"全部轮廓调试图已保存：{all_contours_path}")
    print(f"圆形候选结果已保存：{candidates_path}")
    print(f"候选目标数量：{len(candidates)}")

    for i, candidate in enumerate(candidates, start=1):
        x, y, w, h = candidate["box"]
        area = float(candidate["area"])
        circularity = float(candidate["circularity"])
        print(f"候选 {i}: box=(x={x}, y={y}, w={w}, h={h}), area={area:.1f}, circularity={circularity:.3f}")

    row1 = np.hstack(
        [
            resize_for_display(image),
            resize_for_display(gray_to_bgr(mask)),
        ]
    )
    row2 = np.hstack(
        [
            resize_for_display(all_contours_debug),
            resize_for_display(candidate_result),
        ]
    )
    display = np.vstack([row1, row2])

    # 窗口布局：
    # 左上：原图；右上：黄色 Mask。
    # 左下：全部原始轮廓；右下：通过面积和圆度过滤后的候选目标。
    cv2.imshow("Contour filtering: original | mask | all contours | candidates", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
