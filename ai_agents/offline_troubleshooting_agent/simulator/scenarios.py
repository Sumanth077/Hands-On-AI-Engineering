from models.schemas import FaultType
from simulator.machine import FaultTarget, Machine

FAULT_TARGETS: dict[FaultType, FaultTarget] = {
    "bearing": FaultTarget(
        fault_type="bearing",
        temperature_c=92.0,
        vibration_mm_s=8.0,
        pressure_bar=5.0,
        error_code="BRG-02",
    ),
    "cooling": FaultTarget(
        fault_type="cooling",
        temperature_c=96.0,
        vibration_mm_s=2.3,
        pressure_bar=5.1,
        error_code="TMP-04",
    ),
    "pressure": FaultTarget(
        fault_type="pressure",
        temperature_c=77.0,
        vibration_mm_s=2.4,
        pressure_bar=2.8,
        error_code="PRS-03",
    ),
}


def trigger_fault(machine: Machine, fault_type: FaultType) -> None:
    machine.start_fault(FAULT_TARGETS[fault_type])
