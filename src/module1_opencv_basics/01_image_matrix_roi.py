from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "data" / "images" / "site.jpg"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "module1"


CHANNEL_NAMES = {
    0: "B 蓝色通道",
    1: "G 绿色通道",
    2: "R 红色通道",
}


def load_image(image_path: Path) -> np.ndarray:
    """Read an image with OpenCV and fail clearly if the path is wrong."""
    # cv2.imread(filename, flags)
    # filename: 图片路径字符串。
    # flags 默认是 cv2.IMREAD_COLOR，会以 BGR 三通道彩色图读取。
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(
            f"没有读到图片：{image_path}\n"
            "请先创建 data/images 文件夹，并放入一张名为 site.jpg 的工地图片；\n"
            "或者运行脚本时传入图片路径，例如：\n"
            "python src/module1_opencv_basics/01_image_matrix_roi.py D:/your_image.jpg"
        )

    return image


def apply_roi_channel_zero(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    channel_index: int,
) -> np.ndarray:
    """Set one BGR channel of a rectangular ROI to 0."""
    result = image.copy()

    height, width = result.shape[:2]
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    if x1 >= x2 or y1 >= y2:
        raise ValueError("ROI 坐标不合法，请保证 x1 < x2 且 y1 < y2。")

    if channel_index not in CHANNEL_NAMES:
        raise ValueError("OpenCV 彩色图通道只能是 0、1、2，分别代表 B、G、R。")

    # NumPy 图像矩阵的索引顺序是 image[row, column]，也就是 image[y, x]。
    # 因此矩形区域 ROI 的切片写法是 image[y1:y2, x1:x2]。
    roi = result[y1:y2, x1:x2]

    # roi[:, :, channel_index]
    # 第一个冒号: 选中 ROI 内所有行，也就是所有 y。
    # 第二个冒号: 选中 ROI 内所有列，也就是所有 x。
    # channel_index: 选中 BGR 三个通道中的一个通道。
    roi[:, :, channel_index] = 0

    # cv2.rectangle(img, pt1, pt2, color, thickness)
    # img: 要绘制的图片。
    # pt1: 矩形左上角坐标，格式是 (x1, y1)。
    # pt2: 矩形右下角坐标，格式是 (x2, y2)。
    # color: 线条颜色，OpenCV 使用 BGR 顺序；(0, 255, 0) 表示绿色。
    # thickness: 线条粗细，单位是像素；2 表示 2 像素宽。
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return result


def resize_for_display(image: np.ndarray, max_width: int = 1000) -> np.ndarray:
    """Resize large images so they fit on screen while keeping aspect ratio."""
    height, width = image.shape[:2]

    if width <= max_width:
        return image

    scale = max_width / width
    new_height = int(height * scale)

    # cv2.resize(src, dsize, interpolation)
    # src: 输入图片。
    # dsize: 输出尺寸，格式是 (width, height)，注意这里是宽在前。
    # interpolation: 插值方式；cv2.INTER_AREA 常用于缩小图片，画质较稳定。
    return cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)


def main() -> None:
    import sys

    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
    image = load_image(image_path)

    print(f"图片路径：{image_path}")
    print(f"图片矩阵 shape：{image.shape}")
    print("shape 的含义是：(height, width, channels)")
    print("OpenCV 默认颜色通道顺序是 BGR，不是 RGB。")

    height, width = image.shape[:2]

    # 第一关先用自动坐标做演示。你之后可以把这四个值改成安全帽或人脸的真实坐标。
    x1 = int(width * 0.35)
    y1 = int(height * 0.15)
    x2 = int(width * 0.65)
    y2 = int(height * 0.45)
    channel_index = 2

    result = apply_roi_channel_zero(
        image=image,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        channel_index=channel_index,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "01_roi_red_channel_zero.jpg"

    # cv2.imwrite(filename, img)
    # filename: 输出图片路径。
    # img: 要保存的图像矩阵。
    cv2.imwrite(str(output_path), result)

    print(f"ROI 坐标：x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    print(f"被置零的通道：{channel_index}，也就是 {CHANNEL_NAMES[channel_index]}")
    print(f"结果已保存：{output_path}")
    print("当前示例把 ROI 内的 R 通道置为 0，所以该区域会明显少一些红色成分。")

    display = np.hstack([resize_for_display(image), resize_for_display(result)])

    # cv2.imshow(winname, mat)
    # winname: 窗口标题。
    # mat: 要显示的图像矩阵。
    cv2.imshow("left: original | right: ROI with R channel set to 0", display)

    # cv2.waitKey(delay)
    # delay=0 表示一直等待键盘输入；按任意键后窗口关闭。
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
