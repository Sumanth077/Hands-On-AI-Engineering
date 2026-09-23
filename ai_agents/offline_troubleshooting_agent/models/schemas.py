from datetime import datetime
from typing import Literal

from pydantic import BaseModel

FaultType = Literal["bearing", "cooling", "pressure"]


class MachineReading(BaseModel):
    temperature_c: float
    vibration_mm_s: float
    pressure_bar: float
    status: str
    error_code: str | None
    timestamp: datetime


class Incident(BaseModel):
    incident_id: str
    created_at: datetime
    machine_id: str

    symptoms: list[str]

    temperature_c: float | None
    vibration_mm_s: float | None
    pressure_bar: float | None
    error_code: str | None

    diagnosis: str
    confirmed_cause: str
    fix_applied: str
    outcome: str

    notes: str | None = None
