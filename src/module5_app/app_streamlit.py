# main.py
from __future__ import annotations
import logging
import sys
import time
import os
from pathlib import Path

# ==========================================
# 必须在导入 cv2 和核心逻辑之前设置环境变量
# ==========================================
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

import cv2
cv2.setLogLevel(0)

import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from module5_app.threaded_camera import ThreadedCamera
from module5_app.violation_logger import ViolationLogger

# ==========================================
# 从核心模块导入算法
# ==========================================
from module5_app.vision_core import (
    PersonDetection,
    ComplianceResult,
    detect_persons,
    evaluate,
    draw
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
VIOLATION_DIR = OUTPUT_DIR / "violations"
LOG_DIR = OUTPUT_DIR / "logs"
MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"

st.set_page_config(page_title="Helmet Compliance Dashboard", page_icon=":construction_worker:", layout="wide")

# ==========================================
# 页面 UI 与样式定义
# ==========================================
def inject_modern_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --hc-bg: #0b1020;
            --hc-card: rgba(255, 255, 255, 0.075);
            --hc-card-border: rgba(255, 255, 255, 0.12);
            --hc-text-muted: rgba(255, 255, 255, 0.68);
            --hc-primary: #6ee7b7;
            --hc-danger: #fb7185;
            --hc-warn: #fbbf24;
        }
        .block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1320px; }
        section[data-testid="stSidebar"] { background: linear-gradient(180deg, #111827 0%, #0f172a 100%); border-right: 1px solid rgba(255, 255, 255, 0.08); }
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #f8fafc; }
        .hc-status-row { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 18px 0 8px 0; }
        .hc-status-card { padding: 15px 16px; border-radius: 20px; background: rgba(255, 255, 255, 0.07); border: 1px solid rgba(255, 255, 255, 0.12); }
        .hc-status-label { color: var(--hc-text-muted); font-size: 0.82rem; margin-bottom: 6px; }
        .hc-status-value { color: #ffffff; font-size: 1.1rem; font-weight: 760; }
        .hc-card-title { font-size: 1.3rem; font-weight: 760; margin-top: 1.1rem; margin-bottom: 0.55rem; }
        .hc-mini-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 8px; }
        .hc-mini-card { min-height: 88px; padding: 14px 16px; border-radius: 18px; background: rgba(255, 255, 255, 0.055); border: 1px solid rgba(255, 255, 255, 0.10); box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05); }
        .hc-mini-label { color: var(--hc-text-muted); font-size: 0.78rem; font-weight: 650; margin-bottom: 8px; }
        .hc-mini-value { color: #f8fafc; font-size: 1.55rem; line-height: 1.05; font-weight: 760; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .hc-info-card { margin-top: 14px; padding: 16px 18px; border-radius: 20px; background: linear-gradient(180deg, rgba(15, 23, 42, 0.88), rgba(15, 23, 42, 0.62)); border: 1px solid rgba(148, 163, 184, 0.18); }
        .hc-info-title { color: #f8fafc; font-size: 0.92rem; font-weight: 760; margin-bottom: 10px; }
        .hc-info-line { display: flex; justify-content: space-between; gap: 12px; color: var(--hc-text-muted); font-size: 0.78rem; padding: 5px 0; border-bottom: 1px solid rgba(148, 163, 184, 0.10); }
        .hc-info-line:last-child { border-bottom: none; }
        div[data-testid="stDataFrame"] { border-radius: 18px; overflow: hidden; border: 1px solid rgba(148, 163, 184, 0.20); }
        div[data-testid="stImage"] img { border-radius: 18px; border: 1px solid rgba(148, 163, 184, 0.20); box-shadow: 0 16px 36px rgba(0, 0, 0, 0.18); }
        .stAlert { border-radius: 16px; }
        @media (max-width: 900px) { .hc-status-row { grid-template-columns: repeat(2, minmax(0, 1fr)); } .hc-title { font-size: 1.8rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_system_notes() -> None:
    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True) 
    with st.expander("使用说明与检测规则", expanded=False):
        st.markdown(
            """
            - **开始连续监控**：按左侧设置的刷新间隔持续读取摄像头并检测。
            - **暂停监控**：停止连续刷新和摄像头检测循环。
            - **单帧检测**：只采集并检测当前一帧。
            - **上传图片检测**：不依赖摄像头，直接对本地图片执行检测流程。
            """
        )

# ==========================================
# 状态与资源管理
# ==========================================
def init_state() -> None:
    defaults = {
        "monitoring": False,
        "violation_count": 0,
        "recovery_count": 0,
        "alarm_active": False,
        "last_time": time.time(),
        "total_logs": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

@st.cache_resource
def get_logger() -> ViolationLogger:
    return ViolationLogger(output_dir=OUTPUT_DIR, violation_dir=VIOLATION_DIR, log_dir=LOG_DIR)

@st.cache_resource
def get_camera(index: int) -> ThreadedCamera:
    return ThreadedCamera(index).start()

@st.cache_resource
def get_model() -> YOLO:
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    try:
        from ultralytics.utils import LOGGER
        LOGGER.setLevel(logging.ERROR)
    except Exception:
        pass
    return YOLO(str(MODEL_PATH) if MODEL_PATH.exists() else "yolov8n.pt")

@st.cache_data(ttl=2)
def load_today_df(log_dir: Path) -> pd.DataFrame:
    path = log_dir / f"violations_{pd.Timestamp.now():%Y-%m-%d}.csv"
    cols = ["timestamp", "violation_type", "confidence", "image_path", "frame_index", "person_id", "note"]
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=cols)

@st.cache_data(ttl=2)
def load_images(violation_dir: Path, limit: int = 6) -> list[Path]:
    folder = violation_dir / f"{pd.Timestamp.now():%Y-%m-%d}"
    return sorted(folder.glob("*.jpg"), reverse=True)[:limit] if folder.exists() else []

def rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# ==========================================
# 交互组件与逻辑辅助
# ==========================================
def sidebar() -> dict[str, object]:
    st.sidebar.title("监控控制台")
    col1, col2 = st.sidebar.columns(2)
    if col1.button("开始连续监控", type="primary"):
        st.session_state.monitoring = True
    if col2.button("暂停监控"):
        st.session_state.monitoring = False
    
    single_step = st.sidebar.button("单帧检测")
    
    # ======= 【核心修改 1】：新增专属的弹窗呼出按钮 =======
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    show_logs_btn = st.sidebar.button("📊 近日违规日志和截图", use_container_width=True)
    st.sidebar.markdown("---")
    
    cam = st.sidebar.number_input("摄像头编号", 0, 10, 0, 1)
    interval = st.sidebar.slider("帧间休眠(秒) - 控制CPU占用", 0.0, 1.0, 0.01, 0.01)
    enable = st.sidebar.checkbox("启用 YOLO 检测", value=True)
    
    # 移除了原本的 show_logs checkbox
    
    conf = st.sidebar.slider("YOLO person 置信度", 0.05, 0.95, 0.25, 0.05)
    yellow_th = st.sidebar.slider("头部黄色占比阈值", 0.001, 0.3, 0.03, 0.001)
    v_th = st.sidebar.slider("连续违规报警帧数", 1, 30, 10, 1)
    r_th = st.sidebar.slider("连续正常解除帧数", 1, 10, 3, 1)
    log_on_alarm = st.sidebar.checkbox("报警触发时保存截图和日志", value=True)
    st.sidebar.write("当前状态：" + ("运行中" if st.session_state.monitoring else "已暂停"))
    
    return {
        "active": bool(st.session_state.monitoring or single_step),
        "monitoring": bool(st.session_state.monitoring),
        "cam": int(cam),
        "interval": float(interval),
        "enable": bool(enable),
        "show_logs_btn": show_logs_btn,  # 传递按钮的点击状态
        "conf": float(conf),
        "yellow_th": float(yellow_th),
        "v_th": int(v_th),
        "r_th": int(r_th),
        "log_on_alarm": bool(log_on_alarm),
    }

def update_alarm(is_violation: bool, v_th: int, r_th: int) -> tuple[bool, bool, bool]:
    triggered = cleared = False
    if is_violation:
        st.session_state.violation_count += 1
        st.session_state.recovery_count = 0
        if not st.session_state.alarm_active and st.session_state.violation_count >= v_th:
            st.session_state.alarm_active = True
            triggered = True
    else:
        st.session_state.recovery_count += 1
        st.session_state.violation_count = 0
        if st.session_state.alarm_active and st.session_state.recovery_count >= r_th:
            st.session_state.alarm_active = False
            cleared = True
    return st.session_state.alarm_active, triggered, cleared

def log_alarm(logger: ViolationLogger, vis: np.ndarray, frame_index: int | None, results: list[ComplianceResult], enabled: bool) -> None:
    if not enabled:
        return
    bad = next((r for r in results if not r.compliant), None)
    if bad is None:
        return
    logger.log_violation(vis, bad.violation_type, bad.confidence, frame_index, bad.person_id, f"yellow_ratio={bad.yellow_ratio:.4f}")
    st.session_state.total_logs += 1

def decode_upload(file) -> np.ndarray | None:
    if file is None:
        return None
    data = np.frombuffer(file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

# ==========================================
# 核心视图渲染
# ==========================================
def render_tables(logger: ViolationLogger, show_logs: bool) -> None:
    if show_logs:
        st.markdown('<div class="hc-card-title">今日违规日志</div>', unsafe_allow_html=True)
        df = load_today_df(logger.log_dir)
        st.dataframe(df.sort_values("timestamp", ascending=False) if not df.empty else df, use_container_width=True)
    st.markdown('<div class="hc-card-title">最近违规截图</div>', unsafe_allow_html=True)
    imgs = load_images(logger.violation_dir)
    if not imgs:
        st.info("暂无违规截图。")
        return
    cols = st.columns(min(3, len(imgs)))
    for i, path in enumerate(imgs):
        with cols[i % len(cols)]:
            st.image(str(path), caption=path.name, use_container_width=True)

# ======= 【核心修改 2】：使用 @st.dialog 包装表格渲染 =======
@st.dialog("历史违规数据中心", width="large")
def show_logs_dialog(logger: ViolationLogger):
    # 当弹窗打开时，强行调用 render_tables
    render_tables(logger, show_logs=True)

def render_upload(cfg: dict[str, object], logger: ViolationLogger) -> None:
    st.markdown('<div class="hc-card-title">图片快速检测</div>', unsafe_allow_html=True)
    file = st.file_uploader("拖入或选择一张工地图片", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if file is None:
        return
    image = decode_upload(file)
    if image is None:
        st.error("图片解码失败。")
        return
    c1, c2 = st.columns([0.5, 0.5], gap="large")
    c1.image(rgb(image), caption="原图", use_container_width=True)
    save = st.checkbox("上传图片存在违规时保存截图和日志", value=True)
    if not st.button("检测上传图片", type="primary"):
        return
    
    persons = detect_persons(get_model(), image, float(cfg["conf"]))
    results = evaluate(image, persons, float(cfg["yellow_th"]))
    vis = draw(image, persons, results, 0.0, any(not r.compliant for r in results))
    
    c2.image(rgb(vis), caption="检测结果", use_container_width=True)
    if results:
        st.dataframe(pd.DataFrame([r.__dict__ for r in results]), use_container_width=True)
    bad = [r for r in results if not r.compliant]
    if bad and save:
        log_alarm(logger, vis, None, results, True)
        load_today_df.clear()
        load_images.clear()
        st.success("已保存上传图片违规截图和日志。")

def render_monitor(cfg: dict[str, object], logger: ViolationLogger) -> None:
    if not cfg["active"]:
        st.warning("监控已暂停。点击“开始连续监控”运行，或点击“单帧检测”采集一次。")
        return

    st.markdown('<div class="hc-card-title">实时监控画面</div>', unsafe_allow_html=True)
    alert_placeholder = st.empty()
    video_col, side_col = st.columns([0.72, 0.28], gap="large")
    
    with video_col:
        video_placeholder = st.empty()
    with side_col:
        st.caption("实时概览")
        stats_placeholder = st.empty()
        info_placeholder = st.empty()
        
    st.subheader("当前帧判定结果")
    table_placeholder = st.empty()

    info_placeholder.markdown(
        f'''
        <div class="hc-info-card">
            <div class="hc-info-title">运行摘要</div>
            <div class="hc-info-line"><span>监控模式</span><span>{"连续监控" if cfg["monitoring"] else "单帧检测"}</span></div>
            <div class="hc-info-line"><span>连续违规阈值</span><span>{int(cfg["v_th"])} 帧</span></div>
            <div class="hc-info-line"><span>恢复阈值</span><span>{int(cfg["r_th"])} 帧</span></div>
            <div class="hc-info-line"><span>检测策略</span><span>person + 头部黄色占比</span></div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    camera = get_camera(int(cfg["cam"]))

    while True:
        ok, frame, frame_index = camera.read()
        if not ok or frame is None:

            alert_placeholder.warning("摄像头正在唤醒中，请稍候...")
            time.sleep(0.5) # 稍微等半秒钟让硬件预热
            
            # 如果是单步检测，读不到就算了，跳出
            if not cfg["monitoring"]:
                break
                
            continue # 继续下一轮尝试读取！

        now = time.time()
        fps = 1.0 / max(now - float(st.session_state.last_time), 1e-6)
        st.session_state.last_time = now

        persons: list[PersonDetection] = []
        results: list[ComplianceResult] = []

        if cfg["enable"]:
            persons = detect_persons(get_model(), frame, float(cfg["conf"]))
            results = evaluate(frame, persons, float(cfg["yellow_th"]))

        alarm, triggered, cleared = update_alarm(any(not r.compliant for r in results), int(cfg["v_th"]), int(cfg["r_th"]))
        vis = draw(frame, persons, results, fps, alarm)
        log_alarm(logger, vis, frame_index, results, bool(cfg["log_on_alarm"]) and triggered)

        if triggered: alert_placeholder.error("报警触发，已保存截图和日志。")
        elif cleared: alert_placeholder.success("报警解除。")

        video_placeholder.image(rgb(vis), channels="RGB", use_container_width=True)
        
        stats_placeholder.markdown(
            f'''
            <div class="hc-mini-grid">
                <div class="hc-mini-card"><div class="hc-mini-label">帧编号</div><div class="hc-mini-value">{frame_index}</div></div>
                <div class="hc-mini-card"><div class="hc-mini-label">FPS</div><div class="hc-mini-value">{fps:.1f}</div></div>
                <div class="hc-mini-card"><div class="hc-mini-label">人数</div><div class="hc-mini-value">{len(persons)}</div></div>
                <div class="hc-mini-card"><div class="hc-mini-label">报警</div><div class="hc-mini-value" style="color: {'#fb7185' if alarm else '#6ee7b7'};">{"ACTIVE" if alarm else "NORMAL"}</div></div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        if results:
            table_placeholder.dataframe(pd.DataFrame([r.__dict__ for r in results]), use_container_width=True)
        else:
            table_placeholder.empty()

        if not cfg["monitoring"]: break
        time.sleep(float(cfg["interval"]))

# ==========================================
# 主程序入口
# ==========================================
def main() -> None:
    init_state()
    inject_modern_style()

    cfg = sidebar()
    logger = get_logger()
    
    # ======= 【核心修改 3】：触发逻辑 =======
    # 如果检测到左侧边栏的按钮被点击了，呼出对话框弹窗
    if cfg["show_logs_btn"]:
        show_logs_dialog(logger)
    
    render_upload(cfg, logger)

    render_monitor(cfg, logger)

if __name__ == "__main__":
    main()