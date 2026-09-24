import random
from dataclasses import dataclass
from datetime import datetime

from models.schemas import FaultType, MachineReading
from simulator.history import History

TEMP_RANGE = (65.0, 75.0)
VIBRATION_RANGE = (1.5, 3.0)
PRESSURE_RANGE = (4.5, 5.5)

IDLE_TEMP_JITTER = 0.5
IDLE_VIBRATION_JITTER = 0.1
IDLE_PRESSURE_JITTER = 0.1

FAULT_TEMP_NOISE = 0.3
FAULT_VIBRATION_NOISE = 0.1
FAULT_PRESSURE_NOISE = 0.05


@dataclass(frozen=True)
class FaultTarget:
    fault_type: FaultType
    temperature_c: float
    vibration_mm_s: float
    pressure_bar: float
    error_code: str
    total_steps: int = 8


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _lerp(start: float, end: float, frac: float) -> float:
    return start + (end - start) * frac


class Machine:
    def __init__(self, seed: int | None = None, history_max_length: int = 200):
        self._rng = random.Random(seed)
        self.history = History(max_length=history_max_length)
        self._active_fault: FaultTarget | None = None
        self._fault_start: MachineReading | None = None
        self._fault_step = 0
        self.state: MachineReading
        self.reset_machine()

    def get_current_readings(self) -> MachineReading:
        return self.state

    def reset_machine(self) -> None:
        self._active_fault = None
        self._fault_start = None
        self._fault_step = 0
        self.state = MachineReading(
            temperature_c=self._rng.uniform(*TEMP_RANGE),
            vibration_mm_s=self._rng.uniform(*VIBRATION_RANGE),
            pressure_bar=self._rng.uniform(*PRESSURE_RANGE),
            status="running",
            error_code=None,
            timestamp=datetime.now(),
        )

    def start_fault(self, target: FaultTarget) -> None:
        self._active_fault = target
        self._fault_start = self.state
        self._fault_step = 0

    def advance_simulation(self, steps: int = 1) -> None:
        for _ in range(steps):
            if self._active_fault is not None:
                self._advance_fault_step()
            else:
                self._advance_idle_step()
            self.history.append_reading(self.state)

    def _advance_idle_step(self) -> None:
        temp = _clamp(
            self.state.temperature_c + self._rng.uniform(-IDLE_TEMP_JITTER, IDLE_TEMP_JITTER),
            *TEMP_RANGE,
        )
        vibration = _clamp(
            self.state.vibration_mm_s
            + self._rng.uniform(-IDLE_VIBRATION_JITTER, IDLE_VIBRATION_JITTER),
            *VIBRATION_RANGE,
        )
        pressure = _clamp(
            self.state.pressure_bar
            + self._rng.uniform(-IDLE_PRESSURE_JITTER, IDLE_PRESSURE_JITTER),
            *PRESSURE_RANGE,
        )
        self.state = MachineReading(
            temperature_c=temp,
            vibration_mm_s=vibration,
            pressure_bar=pressure,
            status="running",
            error_code=None,
            timestamp=datetime.now(),
        )

    def _advance_fault_step(self) -> None:
        assert self._active_fault is not None
        assert self._fault_start is not None
        target = self._active_fault
        self._fault_step += 1
        frac = min(self._fault_step / target.total_steps, 1.0)

        temp = _lerp(self._fault_start.temperature_c, target.temperature_c, frac)
        temp += self._rng.uniform(-FAULT_TEMP_NOISE, FAULT_TEMP_NOISE)

        vibration = _lerp(self._fault_start.vibration_mm_s, target.vibration_mm_s, frac)
        vibration = max(0.0, vibration + self._rng.uniform(-FAULT_VIBRATION_NOISE, FAULT_VIBRATION_NOISE))

        pressure = _lerp(self._fault_start.pressure_bar, target.pressure_bar, frac)
        pressure = max(0.0, pressure + self._rng.uniform(-FAULT_PRESSURE_NOISE, FAULT_PRESSURE_NOISE))

        reached_target = frac >= 1.0
        self.state = MachineReading(
            temperature_c=temp,
            vibration_mm_s=vibration,
            pressure_bar=pressure,
            status="fault" if reached_target else "warning",
            error_code=target.error_code if reached_target else None,
            timestamp=datetime.now(),
        )
