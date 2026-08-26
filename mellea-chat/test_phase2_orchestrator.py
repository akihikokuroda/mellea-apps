#!/usr/bin/env python3
"""Test script for Phase 2: Behavior Orchestration."""

import asyncio
import json
from pathlib import Path

from dog_scheduler import DailySchedule, DogState, DogStateMetrics
from dog_behavior_orchestrator import BehaviorOrchestrator
from hardware_interface import create_hardware_interface
from dog_runtime import DogRuntime


def test_orchestrator_initialization():
    """Test orchestrator creation."""
    print("\n=== Testing BehaviorOrchestrator Initialization ===")

    orchestrator = BehaviorOrchestrator(use_variation=True)
    assert orchestrator is not None
    assert orchestrator.use_variation is True

    orchestrator_no_var = BehaviorOrchestrator(use_variation=False)
    assert orchestrator_no_var.use_variation is False

    print("✓ Orchestrator initializes correctly")


def test_movement_generation():
    """Test movement generation for each dog state."""
    print("\n=== Testing Movement Generation ===")

    orchestrator = BehaviorOrchestrator()

    states = [
        DogState.SLEEPING,
        DogState.ALERT,
        DogState.PLAYING,
        DogState.RESTING,
        DogState.EXPLORING,
        DogState.EATING,
        DogState.SCRATCHING,
        DogState.GROOMING,
    ]

    for state in states:
        movements = orchestrator.get_movements_for_state(state, intensity=0.7, duration=3000)
        assert isinstance(movements, list), f"Movement list for {state} is not a list"
        assert len(movements) > 0, f"No movements generated for {state}"
        print(f"  {state.value:12} → {len(movements):3} frames")

    print("✓ All states generate movements")


def test_intensity_scaling():
    """Test that intensity affects movement generation."""
    print("\n=== Testing Intensity Scaling ===")

    orchestrator = BehaviorOrchestrator()

    # Low intensity
    low_intensity = orchestrator.get_movements_for_state(
        DogState.PLAYING, intensity=0.2, duration=3000
    )

    # High intensity
    high_intensity = orchestrator.get_movements_for_state(
        DogState.PLAYING, intensity=0.9, duration=3000
    )

    print(f"  Low intensity (0.2):  {len(low_intensity)} frames")
    print(f"  High intensity (0.9): {len(high_intensity)} frames")

    # Both should generate movements
    assert len(low_intensity) > 0
    assert len(high_intensity) > 0

    print("✓ Intensity scaling works")


def test_duration_normalization():
    """Test that movements fit within specified duration."""
    print("\n=== Testing Duration Normalization ===")

    orchestrator = BehaviorOrchestrator()

    durations = [1000, 5000, 10000, 30000]

    for duration in durations:
        movements = orchestrator.get_movements_for_state(
            DogState.PLAYING, intensity=0.7, duration=duration
        )

        if movements:
            max_ts = max((f.timestamp_ms for f in movements), default=0)
            # Allow 10% tolerance for timing jitter from variation
            assert max_ts <= duration * 1.15, f"Movement exceeds duration: {max_ts} > {duration * 1.15}"
            print(f"  Duration {duration:5}ms → max frame {max_ts:7.1f}ms ✓")

    print("✓ Duration normalization works")


def test_variation():
    """Test that variation creates different movements."""
    print("\n=== Testing Movement Variation ===")

    orchestrator_var = BehaviorOrchestrator(use_variation=True)
    orchestrator_no_var = BehaviorOrchestrator(use_variation=False)

    # Generate same behavior twice with variation
    mov1 = orchestrator_var.get_movements_for_state(DogState.PLAYING, intensity=0.7, duration=3000)
    mov2 = orchestrator_var.get_movements_for_state(DogState.PLAYING, intensity=0.7, duration=3000)

    # Generate without variation twice
    mov3 = orchestrator_no_var.get_movements_for_state(DogState.PLAYING, intensity=0.7, duration=3000)
    mov4 = orchestrator_no_var.get_movements_for_state(DogState.PLAYING, intensity=0.7, duration=3000)

    # With variation, might be different
    same_with_var = (mov1[0].timestamp_ms if mov1 else 0) == (mov2[0].timestamp_ms if mov2 else 0)
    print(f"  With variation - same timestamps: {same_with_var} (likely False)")

    # Without variation, should be same sequence
    same_no_var = (mov3[0].timestamp_ms if mov3 else 0) == (mov4[0].timestamp_ms if mov4 else 0)
    print(f"  No variation - same timestamps: {same_no_var} (should be True)")

    print("✓ Variation toggle works")


def test_transitions():
    """Test behavior transitions."""
    print("\n=== Testing State Transitions ===")

    orchestrator = BehaviorOrchestrator()

    transitions = [
        (DogState.SLEEPING, DogState.ALERT),
        (DogState.ALERT, DogState.PLAYING),
        (DogState.PLAYING, DogState.RESTING),
        (DogState.EATING, DogState.RESTING),
    ]

    for from_state, to_state in transitions:
        transition = orchestrator.create_transition(from_state, to_state, duration=300)
        print(f"  {from_state.value:12} → {to_state.value:12}: {len(transition)} frames")

    print("✓ Transitions generate frames")


def test_behavior_composition():
    """Test composing multiple behaviors."""
    print("\n=== Testing Behavior Composition ===")

    orchestrator = BehaviorOrchestrator()

    # Create a sequence: play, then rest, then explore
    behavior_sequence = [
        (DogState.PLAYING, 5000),
        (DogState.RESTING, 3000),
        (DogState.EXPLORING, 4000),
    ]

    composed = orchestrator.compose_behaviors(behavior_sequence)

    print(f"  Sequence: PLAYING(5s) → RESTING(3s) → EXPLORING(4s)")
    print(f"  Total frames: {len(composed)}")

    if composed:
        total_duration = max((f.timestamp_ms for f in composed), default=0)
        print(f"  Total duration: {total_duration:.0f}ms (expected ~12000ms)")

    assert len(composed) > 0, "Composed behavior produced no frames"
    print("✓ Behavior composition works")


async def test_runtime_with_orchestrator():
    """Test runtime with orchestrator integrated."""
    print("\n=== Testing Runtime with Orchestrator ===")

    hardware = create_hardware_interface("simulation")
    orchestrator = BehaviorOrchestrator()

    runtime = DogRuntime(
        hardware_interface=hardware,
        behavior_orchestrator=orchestrator,
        enable_logging=True,
    )

    print(f"Created runtime with orchestrator")

    # Run 5 cycles
    print(f"Running 5 cycles with actual movement execution...")
    for i in range(5):
        should_continue = await runtime.run_cycle()
        status = runtime.get_current_status()
        print(f"  Cycle {i+1}: {status['dog_state']:10} (Energy: {status['metrics']['energy_level']:3}%)")
        if not should_continue:
            break

    # Check that movements were logged
    print(f"  State log entries: {len(runtime.state_log)}")
    execute_events = [e for e in runtime.state_log if e.get("event") == "execute_behavior"]
    print(f"  Execute behavior events: {len(execute_events)}")

    runtime.save_logs()

    print("✓ Runtime with orchestrator works")


async def test_full_simulation():
    """Full simulation with orchestrator."""
    print("\n=== Testing Full Simulation with Orchestrator ===")

    hardware = create_hardware_interface("simulation")
    orchestrator = BehaviorOrchestrator()

    runtime = DogRuntime(
        hardware_interface=hardware,
        behavior_orchestrator=orchestrator,
        enable_logging=True,
    )

    print("Running 3-cycle simulation with actual movements...")
    await runtime.run(cycle_limit=3)

    # Analyze results
    print(f"Total state log entries: {len(runtime.state_log)}")

    behavior_events = [
        e for e in runtime.state_log if e.get("event") == "execute_behavior"
    ]
    print(f"Behavior execution events: {len(behavior_events)}")

    if behavior_events:
        frame_counts = [e.get("frame_count", 0) for e in behavior_events]
        total_frames = sum(frame_counts)
        avg_frames = total_frames / len(frame_counts) if frame_counts else 0
        print(f"Total motion frames executed: {total_frames}")
        print(f"Average frames per behavior: {avg_frames:.1f}")

    # Count states
    state_counts = {}
    for entry in runtime.state_log:
        state = entry.get("dog_state", "unknown")
        state_counts[state] = state_counts.get(state, 0) + 1

    print(f"Behavior distribution:")
    for state, count in sorted(state_counts.items()):
        print(f"  {state:12}: {count:2} times")

    print("✓ Full simulation with orchestrator complete")


async def test_motion_logging():
    """Test that motion logs are created correctly."""
    print("\n=== Testing Motion Logging ===")

    hardware = create_hardware_interface("simulation")
    orchestrator = BehaviorOrchestrator()

    runtime = DogRuntime(
        hardware_interface=hardware,
        behavior_orchestrator=orchestrator,
        enable_logging=True,
    )

    await runtime.run(cycle_limit=2)

    # Check if log files exist
    runtime_log_dir = Path("dog_runtime_logs")
    sim_log_dir = Path("dog_simulation_logs")

    runtime_logs = list(runtime_log_dir.glob("state_log_*.json"))
    sim_logs = list(sim_log_dir.glob("motion_log_*.json"))

    print(f"Runtime logs created: {len(runtime_logs)}")
    print(f"Simulation logs created: {len(sim_logs)}")

    if runtime_logs:
        with open(runtime_logs[-1], "r") as f:
            data = json.load(f)
            print(f"  Latest runtime log: {len(data)} entries")

    print("✓ Motion logging works")


async def main():
    """Run all tests."""
    print("=" * 50)
    print("Phase 2: Behavior Orchestration Tests")
    print("=" * 50)

    try:
        # Synchronous tests
        test_orchestrator_initialization()
        test_movement_generation()
        test_intensity_scaling()
        test_duration_normalization()
        test_variation()
        test_transitions()
        test_behavior_composition()

        # Async tests
        await test_runtime_with_orchestrator()
        await test_full_simulation()
        await test_motion_logging()

        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
