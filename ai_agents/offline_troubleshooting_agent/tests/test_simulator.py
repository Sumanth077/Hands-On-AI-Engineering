import pytest

from simulator.machine import (
    Machine,
    PRESSURE_RANGE,
    TEMP_RANGE,
    VIBRATION_RANGE,
)
from simulator.scenarios import FAULT_TARGETS, trigger_fault


def test_idle_readings_stay_within_baseline_ranges():
    machine = Machine(seed=42)

    for _ in range(200):
        machine.advance_simulation()
        reading = machine.get_current_readings()
        assert TEMP_RANGE[0] <= reading.temperature_c <= TEMP_RANGE[1]
        assert VIBRATION_RANGE[0] <= reading.vibration_mm_s <= VIBRATION_RANGE[1]
        assert PRESSURE_RANGE[0] <= reading.pressure_bar <= PRESSURE_RANGE[1]
        assert reading.status == "running"
        assert reading.error_code is None

    history = machine.history.get_recent_history(limit=200)
    assert len(history) == 200


@pytest.mark.parametrize(
    "fault_type,error_code",
    [("bearing", "BRG-02"), ("cooling", "TMP-04"), ("pressure", "PRS-03")],
)
def test_fault_scenarios_reach_intended_error_code(fault_type, error_code):
    machine = Machine(seed=7)
    trigger_fault(machine, fault_type)
    machine.advance_simulation(steps=FAULT_TARGETS[fault_type].total_steps + 5)

    reading = machine.get_current_readings()
    assert reading.error_code == error_code
    assert reading.status == "fault"


def test_bearing_fault_pattern():
    machine = Machine(seed=7)
    trigger_fault(machine, "bearing")
    machine.advance_simulation(steps=FAULT_TARGETS["bearing"].total_steps + 5)

    reading = machine.get_current_readings()
    assert reading.temperature_c > TEMP_RANGE[1]
    assert reading.vibration_mm_s > VIBRATION_RANGE[1] * 2
    assert PRESSURE_RANGE[0] - 0.5 <= reading.pressure_bar <= PRESSURE_RANGE[1] + 0.5


def test_cooling_fault_pattern():
    machine = Machine(seed=7)
    trigger_fault(machine, "cooling")
    machine.advance_simulation(steps=FAULT_TARGETS["cooling"].total_steps + 5)

    reading = machine.get_current_readings()
    assert reading.temperature_c > TEMP_RANGE[1]
    assert VIBRATION_RANGE[0] - 0.5 <= reading.vibration_mm_s <= VIBRATION_RANGE[1] + 0.5
    assert PRESSURE_RANGE[0] - 0.5 <= reading.pressure_bar <= PRESSURE_RANGE[1] + 0.5


def test_pressure_fault_pattern():
    machine = Machine(seed=7)
    trigger_fault(machine, "pressure")
    machine.advance_simulation(steps=FAULT_TARGETS["pressure"].total_steps + 5)

    reading = machine.get_current_readings()
    assert reading.pressure_bar < PRESSURE_RANGE[0]
    assert reading.temperature_c > TEMP_RANGE[1]
    assert VIBRATION_RANGE[0] - 0.5 <= reading.vibration_mm_s <= VIBRATION_RANGE[1] + 0.5


def test_reset_machine_returns_to_normal_baseline_after_fault():
    machine = Machine(seed=3)
    trigger_fault(machine, "bearing")
    machine.advance_simulation(steps=FAULT_TARGETS["bearing"].total_steps + 5)
    assert machine.get_current_readings().error_code == "BRG-02"

    machine.reset_machine()
    reading = machine.get_current_readings()
    assert TEMP_RANGE[0] <= reading.temperature_c <= TEMP_RANGE[1]
    assert VIBRATION_RANGE[0] <= reading.vibration_mm_s <= VIBRATION_RANGE[1]
    assert PRESSURE_RANGE[0] <= reading.pressure_bar <= PRESSURE_RANGE[1]
    assert reading.status == "running"
    assert reading.error_code is None

    machine.advance_simulation(steps=5)
    reading = machine.get_current_readings()
    assert reading.error_code is None
    assert reading.status == "running"
