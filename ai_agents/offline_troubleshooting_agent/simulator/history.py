import pandas as pd

from models.schemas import MachineReading

READING_COLUMNS = [
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "status",
    "error_code",
    "timestamp",
]


class History:
    def __init__(self, max_length: int = 200):
        self._max_length = max_length
        self._df = pd.DataFrame(columns=READING_COLUMNS)

    def append_reading(self, reading: MachineReading) -> None:
        row = pd.DataFrame([reading.model_dump()])
        if self._df.empty:
            self._df = row
        else:
            self._df = pd.concat([self._df, row], ignore_index=True)
        if len(self._df) > self._max_length:
            self._df = self._df.iloc[-self._max_length :].reset_index(drop=True)

    def get_recent_history(self, limit: int = 20) -> pd.DataFrame:
        return self._df.tail(limit).reset_index(drop=True)
