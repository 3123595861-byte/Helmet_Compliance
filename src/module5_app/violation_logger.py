from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any
import csv

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_VIOLATION_DIR = DEFAULT_OUTPUT_DIR / "violations"
DEFAULT_LOG_DIR = DEFAULT_OUTPUT_DIR / "logs"


@dataclass
class ViolationRecord:
    """One violation entry for CSV logging."""

    timestamp: str
    violation_type: str
    confidence: float
    image_path: str
    frame_index: int | None = None
    person_id: int | None = None
    note: str = ""

    def to_row(self) -> dict[str, Any]:
        """Convert the record to a CSV-friendly row."""
        row = asdict(self)
        row["confidence"] = f"{self.confidence:.4f}"
        row["frame_index"] = "" if self.frame_index is None else str(self.frame_index)
        row["person_id"] = "" if self.person_id is None else str(self.person_id)
        return row


class ViolationLogger:
    """Save violation screenshots and append structured CSV logs."""

    def __init__(
        self,
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
        violation_dir: Path | str = DEFAULT_VIOLATION_DIR,
        log_dir: Path | str = DEFAULT_LOG_DIR,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.violation_dir = Path(violation_dir)
        self.log_dir = Path(log_dir)
        self.violation_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _today_stamp() -> str:
        return datetime.now().strftime("%Y-%m-%d")

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def _date_folder(self) -> Path:
        folder = self.violation_dir / self._today_stamp()
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _csv_path(self) -> Path:
        return self.log_dir / f"violations_{self._today_stamp()}.csv"

    def save_screenshot(
        self,
        frame_bgr: np.ndarray,
        violation_type: str,
        frame_index: int | None = None,
        person_id: int | None = None,
        confidence: float | None = None,
        prefix: str = "violation",
    ) -> Path:
        """Save a violation screenshot to a dated folder."""
        timestamp = self._timestamp()
        confidence_text = "na" if confidence is None else f"{confidence:.2f}"
        frame_text = "na" if frame_index is None else f"frame_{frame_index}"
        person_text = "na" if person_id is None else f"person_{person_id}"

        filename = f"{timestamp}_{prefix}_{violation_type}_{frame_text}_{person_text}_{confidence_text}.jpg"
        output_path = self._date_folder() / filename

        # cv2.imwrite(filename, img)
        # filename: 图片保存路径。
        # img: 要保存的 BGR 图像矩阵。
        cv2.imwrite(str(output_path), frame_bgr)
        return output_path

    def append_record(self, record: ViolationRecord) -> Path:
        """Append one record to the daily CSV file."""
        csv_path = self._csv_path()
        is_new_file = not csv_path.exists()

        # newline="" 是 csv 模块在 Windows 下推荐的写法，避免空行问题。
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "violation_type",
                    "confidence",
                    "image_path",
                    "frame_index",
                    "person_id",
                    "note",
                ],
            )
            if is_new_file:
                writer.writeheader()
            writer.writerow(record.to_row())

        return csv_path

    def log_violation(
        self,
        frame_bgr: np.ndarray,
        violation_type: str,
        confidence: float = 0.0,
        frame_index: int | None = None,
        person_id: int | None = None,
        note: str = "",
        prefix: str = "violation",
    ) -> tuple[Path, Path, ViolationRecord]:
        """Save screenshot and CSV record together."""
        image_path = self.save_screenshot(
            frame_bgr=frame_bgr,
            violation_type=violation_type,
            frame_index=frame_index,
            person_id=person_id,
            confidence=confidence,
            prefix=prefix,
        )

        record = ViolationRecord(
            timestamp=self._timestamp(),
            violation_type=violation_type,
            confidence=confidence,
            image_path=str(image_path),
            frame_index=frame_index,
            person_id=person_id,
            note=note,
        )
        csv_path = self.append_record(record)
        return image_path, csv_path, record

    def load_recent_records(self, limit: int = 20) -> list[dict[str, str]]:
        """Load the most recent records from today's CSV file."""
        csv_path = self._csv_path()
        if not csv_path.exists():
            return []

        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        return rows[-limit:]

    def list_violation_images(self) -> list[Path]:
        """List all saved violation screenshots for today, newest first."""
        folder = self._date_folder()
        images = sorted(folder.glob("*.jpg"), reverse=True)
        return images


if __name__ == "__main__":
    logger = ViolationLogger()

    # 创建一张简单的测试图，验证保存和日志功能。
    test_frame = np.full((240, 360, 3), 255, dtype=np.uint8)
    cv2.putText(test_frame, "Violation demo", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    image_path, csv_path, record = logger.log_violation(
        frame_bgr=test_frame,
        violation_type="no_helmet",
        confidence=0.97,
        frame_index=12,
        person_id=1,
        note="demo record",
    )

    print("违规截图已保存：", image_path)
    print("CSV 已更新：", csv_path)
    print("记录内容：", record)
