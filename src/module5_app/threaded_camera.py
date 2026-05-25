from __future__ import annotations

from threading import Lock, Thread
from time import sleep

import cv2
import numpy as np


class ThreadedCamera:
    """Read frames from a camera in a background thread."""

    def __init__(self, source: int | str = 0, name: str = "ThreadedCamera") -> None:
        # cv2.VideoCapture(source)
        # source: 摄像头编号或视频文件路径。
        self.source = source
        self.name = name
        self.capture = cv2.VideoCapture(source)
        self._lock = Lock()
        self._running = False
        self._thread: Thread | None = None
        self._frame: np.ndarray | None = None
        self._ret: bool = False
        self._frame_index: int = 0

        if not self.capture.isOpened():
            raise RuntimeError(f"无法打开视频源：{source}")

    @property
    def frame_index(self) -> int:
        """Return how many frames have been captured so far."""
        with self._lock:
            return self._frame_index

    def start(self) -> ThreadedCamera:
        """Start the background frame-grabbing thread."""
        if self._running:
            return self

        self._running = True
        self._thread = Thread(target=self._update, name=self.name, daemon=True)
        self._thread.start()
        return self

    def _update(self) -> None:
        """Continuously grab the latest frame from the source."""
        while self._running:
            ret, frame = self.capture.read()

            with self._lock:
                self._ret = ret
                if ret:
                    self._frame = frame
                    self._frame_index += 1

            if not ret:
                # 视频源临时没有读到帧时稍微让出 CPU。
                sleep(0.01)

    def read(self) -> tuple[bool, np.ndarray | None, int]:
        """Return the latest frame, a success flag, and a frame index."""
        with self._lock:
            if self._frame is None:
                return False, None, self._frame_index

            return self._ret, self._frame.copy(), self._frame_index

    def stop(self) -> None:
        """Stop the background thread and release the capture."""
        self._running = False

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if self.capture.isOpened():
            self.capture.release()

    def is_running(self) -> bool:
        """Check whether the background reader is running."""
        return self._running

    def __enter__(self) -> ThreadedCamera:
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


if __name__ == "__main__":
    camera = ThreadedCamera(0).start()
    try:
        while True:
            ret, frame, index = camera.read()
            if not ret or frame is None:
                continue

            cv2.putText(
                frame,
                f"Frame: {index}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Threaded Camera Demo", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()
