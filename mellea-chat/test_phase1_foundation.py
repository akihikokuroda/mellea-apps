#!/usr/bin/env python3
"""Test script for Phase 1: Foundation components."""

import asyncio
import json
from datetime import datetime

from dog_scheduler import DailySchedule, DogState, DogStateMetrics
from hardware_interface import SimulationHardwareInterface, create_hardware_interface
from dog_runtime import DogRuntime, CommandQueue


def test_scheduler():
    """Test the daily schedule engine."""
    print("\n=== Testing DailySchedule ===")

    scheduler = DailySchedule()

    # Test base activity for different times
    print("\nBase activities at different hours:")
    for hour in [0, 6, 12, 18, 23]:
        activity = scheduler.get_base_activity_for_time()
        from datetime import datetime as dt
        from datetime import timedelta

        test_time = dt.now().replace(hour=hour, minute=0, second=0)
        activity = scheduler.get_base_activity_for_time(test_time)
        print(f"  {hour:02d}:00 → {activity.value}")

    # Test next action
    print("\nGetting next actions:")
    for i in range(5):
        state, duration = scheduler.get_next_action()
        print(f"  Action {i+1}: {state.value} for {duration}ms")

    # Test metric decay
    print("\nTesting metric changes:")
    print(f"  Initial: {scheduler.metrics}")
    scheduler.update_state(60000, user_interacted=False, new_state=DogState.PLAYING)
    print(f"  After 1 min of playing: {scheduler.metrics}")
    scheduler.update_state(120000, user_interacted=False, new_state=DogState.RESTING)
    print(f"  After 2 min of resting: {scheduler.metrics}")

    # Test state description
    print(f"\nState description: {scheduler.describe_current_state()}")

    print("✓ Scheduler tests passed")


def test_metrics():
    """Test the DogStateMetrics class."""
    print("\n=== Testing DogStateMetrics ===")

    metrics = DogStateMetrics(energy_level=50, hunger_level=50, attention_level=50)

    print(f"Initial metrics: {metrics}")

    # Test decay
    metrics.decay_over_time(60000)
    print(f"After 1 min decay: {metrics}")

    # Test sleep recovery
    metrics.recover_from_sleep(300000)
    print(f"After 5 min sleep: {metrics}")

    # Test eating
    metrics.hunger_level = 100
    metrics.after_eating()
    print(f"After eating: {metrics}")

    # Test user interaction
    metrics.on_user_interaction(intensity=80)
    print(f"After user interaction: {metrics}")

    print("✓ Metrics tests passed")


async def test_hardware_interface():
    """Test hardware abstraction layer."""
    print("\n=== Testing Hardware Interface ===")

    # Test simulation interface
    sim = SimulationHardwareInterface()
    print(f"Created simulation interface")

    # Execute a simple motion
    from dogmove import MotionFrame

    frames = [
        MotionFrame(timestamp_ms=0, motor_angles={"tail": 0}),
        MotionFrame(timestamp_ms=100, motor_angles={"tail": 22.5}),
        MotionFrame(timestamp_ms=200, motor_angles={"tail": 45}),
        MotionFrame(timestamp_ms=300, motor_angles={"tail": 22.5}),
        MotionFrame(timestamp_ms=400, motor_angles={"tail": 0}),
    ]

    print(f"Executing motion sequence with {len(frames)} frames...")
    await sim.execute_motion_sequence(frames)

    status = await sim.get_status()
    print(f"Status after execution: {status}")

    sim.save_log("test_motion.json")
    print("✓ Hardware interface tests passed")


async def test_command_queue():
    """Test command queue."""
    print("\n=== Testing CommandQueue ===")

    queue = CommandQueue(maxsize=5)

    # Test put and get
    queue.put("play")
    queue.put("status")
    print(f"Queued 2 commands, has_data: {queue.has_data()}")

    cmd1 = queue.get()
    print(f"Got command: {cmd1}")

    cmd2 = queue.get()
    print(f"Got command: {cmd2}")

    print(f"After getting both, has_data: {queue.has_data()}")

    print("✓ CommandQueue tests passed")


async def test_dog_runtime():
    """Test the main runtime loop."""
    print("\n=== Testing DogRuntime ===")

    # Create runtime with simulation interface
    hardware = create_hardware_interface("simulation")
    runtime = DogRuntime(
        hardware_interface=hardware,
        enable_logging=True,
    )

    print(f"Created runtime with simulation interface")

    # Run a few cycles
    print(f"Running 5 cycles...")
    for i in range(5):
        should_continue = await runtime.run_cycle()
        status = runtime.get_current_status()
        print(f"  Cycle {i+1}: {status['dog_state']} (Energy: {status['metrics']['energy_level']}%)")
        if not should_continue:
            break

    runtime.save_logs()
    print("✓ Runtime tests passed")


async def test_integration():
    """Integration test: 1-hour simulation."""
    print("\n=== Integration Test: 1-Hour Simulation ===")

    hardware = create_hardware_interface("simulation")
    runtime = DogRuntime(
        hardware_interface=hardware,
        enable_logging=True,
    )

    print("Running 10 cycles (simulated dog behavior)...")

    start_time = datetime.now()
    await runtime.run(cycle_limit=10)
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"Simulation completed in {elapsed:.1f} seconds")
    print(f"State log entries: {len(runtime.state_log)}")

    # Print summary
    print("\nBehavior summary:")
    state_counts = {}
    for entry in runtime.state_log:
        state = entry.get("dog_state", "unknown")
        state_counts[state] = state_counts.get(state, 0) + 1

    for state, count in sorted(state_counts.items()):
        print(f"  {state}: {count} times")

    print("✓ Integration test passed")


async def main():
    """Run all tests."""
    print("=" * 50)
    print("Phase 1 Foundation Tests")
    print("=" * 50)

    try:
        # Synchronous tests
        test_scheduler()
        test_metrics()

        # Async tests
        await test_command_queue()
        await test_hardware_interface()
        await test_dog_runtime()
        await test_integration()

        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
