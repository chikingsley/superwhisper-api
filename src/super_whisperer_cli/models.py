from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path  # noqa: TC003


@dataclass(frozen=True)
class Transcript:
    audio_path: Path
    recording_rowid: int
    recording_id: str
    datetime: str
    folder_name: str
    model_name: str
    mode_name: str
    duration: float | None
    processing_time: int | None
    transcript: str

    def as_dict(self) -> dict[str, object]:
        return {
            "audio_path": str(self.audio_path),
            "recording_rowid": self.recording_rowid,
            "recording_id": self.recording_id,
            "datetime": self.datetime,
            "folder_name": self.folder_name,
            "model_name": self.model_name,
            "mode_name": self.mode_name,
            "duration": self.duration,
            "processing_time": self.processing_time,
            "transcript": self.transcript,
        }
