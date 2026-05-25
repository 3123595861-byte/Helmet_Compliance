# Helmet Compliance

**中文**：一个基于 **OpenCV + YOLOv8 + Streamlit** 的安全帽合规检测项目，支持摄像头实时监控、单帧检测、图片上传检测，以及违规截图与 CSV 日志保存。

**English**: A **OpenCV + YOLOv8 + Streamlit** based helmet compliance detection project with real-time camera monitoring, single-frame inspection, image upload detection, and violation screenshot/CSV logging.

---

## Features

- **Real-time monitoring / 实时监控**: continuous camera-based inspection
- **Single-frame detection / 单帧检测**: inspect one frame on demand
- **Image upload detection / 图片检测**: run detection without a camera
- **Person detection / 人员检测**: YOLOv8-based person localization
- **Helmet compliance check / 安全帽判定**: estimate the head region and check yellow pixel ratio
- **Violation logging / 违规留痕**: save screenshots and daily CSV logs automatically
- **Streamlit dashboard / 可视化面板**: clean and interactive UI

---

## Tech Stack

- Python 3.12+
- OpenCV
- Ultralytics YOLOv8
- Streamlit
- NumPy
- Pandas

---

## Project Structure

```text
Helmet_Compliance/
├── requirements.txt
├── src/
│   └── module5_app/
│       ├── app_streamlit.py
│       ├── vision_core.py
│       ├── threaded_camera.py
│       └── violation_logger.py
└── outputs/
    ├── violations/
    └── logs/
```

---

## Installation / 安装

```bash
pip install -r requirements.txt
```

---

## Run / 运行

```bash
streamlit run src/module5_app/app_streamlit.py
```

---

## Usage / 使用说明

**中文**

- 点击 **开始连续监控**：持续检测摄像头画面
- 点击 **暂停监控**：停止连续监控
- 点击 **单帧检测**：仅检测当前一帧
- 上传图片后可直接执行检测
- 可在侧边栏调整置信度、阈值和摄像头编号

**English**

- Click **Start Monitoring** to keep checking the camera stream
- Click **Pause Monitoring** to stop continuous detection
- Click **Single-frame Detection** to inspect the current frame once
- Upload an image to run offline detection
- Adjust confidence, thresholds, and camera index in the sidebar

---

## Output / 输出

- `outputs/violations/YYYY-MM-DD/` — violation screenshots / 违规截图
- `outputs/logs/violations_YYYY-MM-DD.csv` — daily CSV logs / 每日 CSV 日志

---

## Detection Logic / 检测逻辑

1. Detect persons with YOLOv8 / 使用 YOLOv8 检测人员
2. Estimate the head region / 根据人体框估算头部区域
3. Check yellow pixel ratio in the head area / 统计头部区域黄色像素占比
4. Mark as compliant or violation / 判断是否合规
5. Trigger alarm after consecutive violations / 连续违规后触发报警

---

## Notes / 说明

- This project uses a **yellow-region heuristic** for helmet detection, which is suitable for demos and prototypes.
- For production use, a dedicated helmet detection model is recommended.
- The default model file is `yolov8n.pt`. If it is not found locally, Ultralytics will attempt to load/download it.

---

