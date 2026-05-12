from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Event:
    event_id: int
    event_type: str  # "scale_row" | "add_scaled_row" | "swap_rows"
    params: Dict[str, Any]
    timestamp: float
    step: int

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "params": self.params,
            "timestamp": self.timestamp,
            "step": self.step,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(
            event_id=d["event_id"],
            event_type=d["event_type"],
            params=d["params"],
            timestamp=d["timestamp"],
            step=d["step"],
        )
