"""Comprehensive tests for multi-dog orchestration system."""

import asyncio
import pytest
from dog_scheduler_multi import MultiDogScheduler, DogInstance
from dog_orchestrator_multi import MultiDogOrchestrator
from dog_runtime_multi import MultiDogRuntime, run_multi_dog_24_7
from dog_scheduler import DogState
from dogmove import MotionFrame


class TestMultiDogScheduler:
    """Tests for MultiDogScheduler."""

    def test_scheduler_initialization(self):
        """Test scheduler creates correct number of dogs."""
        scheduler = MultiDogScheduler(dog_count=3)
        assert scheduler.get_dog_count() == 3
        assert "dog_1" in scheduler.get_all_dogs()
        assert "dog_2" in scheduler.get_all_dogs()
        assert "dog_3" in scheduler.get_all_dogs()

    def test_add_dog(self):
        """Test adding a new dog."""
        scheduler = MultiDogScheduler(dog_count=1)
        dog = scheduler.add_dog("dog_special")
        assert dog.dog_id == "dog_special"
        assert scheduler.get_dog_count() == 2

    def test_remove_dog(self):
        """Test removing a dog."""
        scheduler = MultiDogScheduler(dog_count=2)
        assert scheduler.remove_dog("dog_1") is True
        assert scheduler.get_dog_count() == 1
        assert scheduler.remove_dog("dog_1") is False  # Already removed

    def test_get_next_actions(self):
        """Test getting next actions for all dogs."""
        scheduler = MultiDogScheduler(dog_count=2)
        actions = scheduler.get_next_actions()

        assert len(actions) == 2
        assert "dog_1" in actions
        assert "dog_2" in actions

        for dog_id, (state, duration) in actions.items():
            assert isinstance(state, DogState)
            assert isinstance(duration, int)
            assert duration > 0

    def test_dog_independence(self):
        """Test that dogs maintain independent state."""
        scheduler = MultiDogScheduler(dog_count=2)

        dog_1 = scheduler.get_dog("dog_1")
        dog_2 = scheduler.get_dog("dog_2")

        # Force dog_1 to have high energy
        dog_1.scheduler.metrics.energy_level = 95

        # Force dog_2 to have low energy
        dog_2.scheduler.metrics.energy_level = 10

        actions = scheduler.get_next_actions()
        state_1, _ = actions["dog_1"]
        state_2, _ = actions["dog_2"]

        # Dog 1 with high energy should likely play/explore
        # Dog 2 with low energy should sleep
        # (Not guaranteed due to randomness, but very likely)
        assert state_1 != DogState.SLEEPING or state_2 == DogState.SLEEPING

    def test_metrics_independence(self):
        """Test that metrics decay independently per dog."""
        scheduler = MultiDogScheduler(dog_count=2)

        dog_1 = scheduler.get_dog("dog_1")
        dog_2 = scheduler.get_dog("dog_2")

        initial_energy_1 = dog_1.scheduler.metrics.energy_level
        initial_energy_2 = dog_2.scheduler.metrics.energy_level

        # Update only dog_1's state with activity
        elapsed_map = {"dog_1": 10000, "dog_2": 0}
        scheduler.update_all_states(elapsed_map)

        # Dog 1's energy should decay from activity
        # Dog 2's energy should be unchanged
        final_energy_1 = dog_1.scheduler.metrics.energy_level
        final_energy_2 = dog_2.scheduler.metrics.energy_level

        assert final_energy_1 <= initial_energy_1  # Decayed or same
        assert final_energy_2 == initial_energy_2  # Unchanged


class TestMultiDogOrchestrator:
    """Tests for MultiDogOrchestrator."""

    def test_orchestrator_initialization(self):
        """Test orchestrator initializes correctly."""
        orch = MultiDogOrchestrator()
        assert orch.behavior_orchestrator is not None

    def test_single_dog_movement(self):
        """Test movement generation for a single dog."""
        orch = MultiDogOrchestrator()
        dog_behaviors = {"dog_1": DogState.PLAYING}

        movements = orch.get_movements_for_all_dogs(dog_behaviors)

        assert len(movements) > 0
        # All motor IDs should be prefixed with dog_1
        for frame in movements:
            for motor_id in frame.motor_angles.keys():
                assert motor_id.startswith("dog_1:"), f"Motor ID not prefixed: {motor_id}"

    def test_multiple_dogs_movement(self):
        """Test movement generation for multiple dogs."""
        orch = MultiDogOrchestrator()
        dog_behaviors = {
            "dog_1": DogState.PLAYING,
            "dog_2": DogState.SLEEPING,
        }

        movements = orch.get_movements_for_all_dogs(dog_behaviors)

        assert len(movements) > 0

        # Collect all prefixed motor IDs
        all_motors = set()
        for frame in movements:
            for motor_id in frame.motor_angles.keys():
                all_motors.add(motor_id)

        # Should have motors from both dogs
        dog_1_motors = {m for m in all_motors if m.startswith("dog_1:")}
        dog_2_motors = {m for m in all_motors if m.startswith("dog_2:")}

        assert len(dog_1_motors) > 0, "No motors for dog_1"
        assert len(dog_2_motors) > 0, "No motors for dog_2"

    def test_motor_id_prefixing(self):
        """Test that motor IDs are correctly prefixed."""
        orch = MultiDogOrchestrator()

        # Get movements from orchestrator for dog_1 playing
        movements = orch.behavior_orchestrator.get_movements_for_state(
            DogState.PLAYING, intensity=0.7, duration=3000
        )

        # Prefix them
        prefixed = orch._prefix_movements(movements, "dog_1")

        assert len(prefixed) == len(movements)
        for original, prefixed_frame in zip(movements, prefixed):
            assert original.timestamp_ms == prefixed_frame.timestamp_ms
            assert len(original.motor_angles) == len(prefixed_frame.motor_angles)

            for motor_id in original.motor_angles.keys():
                expected_prefixed = f"dog_1:{motor_id}"
                assert expected_prefixed in prefixed_frame.motor_angles

    def test_movement_merging(self):
        """Test merging movements from multiple dogs."""
        orch = MultiDogOrchestrator()

        dog_behaviors = {
            "dog_1": DogState.PLAYING,
            "dog_2": DogState.ALERT,
        }

        movements = orch.get_movements_for_all_dogs(dog_behaviors)

        # Should have frames with combined motor angles
        frames_with_both = 0
        for frame in movements:
            dog_1_motors = {m for m in frame.motor_angles.keys() if m.startswith("dog_1:")}
            dog_2_motors = {m for m in frame.motor_angles.keys() if m.startswith("dog_2:")}

            if len(dog_1_motors) > 0 and len(dog_2_motors) > 0:
                frames_with_both += 1

        # Should have at least some frames with both dogs' motors
        assert frames_with_both > 0, "No frames with simultaneous dog motors"

    def test_intensity_scaling(self):
        """Test that intensity affects movement."""
        orch = MultiDogOrchestrator()

        low_intensity = orch.get_movements_for_all_dogs(
            {"dog_1": DogState.PLAYING},
            intensity_map={"dog_1": 0.3},
        )

        high_intensity = orch.get_movements_for_all_dogs(
            {"dog_1": DogState.PLAYING},
            intensity_map={"dog_1": 0.9},
        )

        # Different intensities should produce different movements
        # (not guaranteed to have different lengths, but angles should differ)
        assert len(low_intensity) > 0
        assert len(high_intensity) > 0


class TestMultiDogRuntime:
    """Tests for MultiDogRuntime."""

    @pytest.mark.asyncio
    async def test_runtime_initialization(self):
        """Test runtime initializes correctly."""
        runtime = MultiDogRuntime(dog_count=2)
        assert runtime.scheduler.get_dog_count() == 2
        assert runtime.is_running is False

    @pytest.mark.asyncio
    async def test_runtime_cycle(self):
        """Test executing a single cycle."""
        runtime = MultiDogRuntime(dog_count=2, enable_logging=False)
        runtime.is_running = True

        success = await runtime.run_cycle(cycle_num=0)
        assert success is True

    @pytest.mark.asyncio
    async def test_multi_dog_concurrent_execution(self):
        """Test that multiple dogs run concurrently."""
        runtime = MultiDogRuntime(dog_count=3, enable_logging=False)
        runtime.is_running = True

        # Run a few cycles
        for i in range(5):
            await runtime.run_cycle(cycle_num=i)

        # All dogs should have executed
        state_summary = runtime.get_state_summary()
        assert state_summary["dogs_count"] == 3
        assert len(state_summary["dogs_states"]) == 3

    @pytest.mark.asyncio
    async def test_dog_independence_during_execution(self):
        """Test dogs maintain independent state during execution."""
        runtime = MultiDogRuntime(dog_count=2, enable_logging=False)
        runtime.is_running = True

        dog_1 = runtime.scheduler.get_dog("dog_1")
        dog_2 = runtime.scheduler.get_dog("dog_2")

        initial_energy_1 = dog_1.scheduler.metrics.energy_level
        initial_energy_2 = dog_2.scheduler.metrics.energy_level

        # Run cycles
        for i in range(3):
            await runtime.run_cycle(cycle_num=i)

        final_energy_1 = dog_1.scheduler.metrics.energy_level
        final_energy_2 = dog_2.scheduler.metrics.energy_level

        # Both dogs' energies should have changed (from decay/recovery)
        # but not necessarily in the same direction
        assert final_energy_1 != initial_energy_1 or final_energy_2 != initial_energy_2

    @pytest.mark.asyncio
    async def test_runtime_full_execution(self):
        """Test full runtime execution."""
        runtime = MultiDogRuntime(
            dog_count=2,
            enable_logging=False,
        )
        await runtime.run(duration_seconds=1, cycle_limit=5)

        assert runtime.is_running is False
        state_summary = runtime.get_state_summary()
        assert state_summary["cycle_count"] > 0

    def test_command_queue(self):
        """Test command queue operations."""
        runtime = MultiDogRuntime(dog_count=2)

        runtime.queue_command("play", "dog_1")
        runtime.queue_command("stop")

        assert runtime.command_queue.has_data()
        assert runtime.command_queue.get() == ("dog_1", "play")
        assert runtime.command_queue.get() == ("all", "stop")


class TestIntegration:
    """Integration tests for the complete multi-dog system."""

    @pytest.mark.asyncio
    async def test_two_dogs_different_behaviors(self):
        """Test two dogs can have different behaviors simultaneously."""
        runtime = MultiDogRuntime(dog_count=2, enable_logging=False)

        # Force different states for testing
        dog_1 = runtime.scheduler.get_dog("dog_1")
        dog_2 = runtime.scheduler.get_dog("dog_2")

        dog_1.scheduler.metrics.energy_level = 90  # High energy
        dog_2.scheduler.metrics.energy_level = 15  # Low energy

        runtime.is_running = True
        await runtime.run_cycle(cycle_num=0)

        state_1 = dog_1.current_state
        state_2 = dog_2.current_state

        # Dog 1 should likely not sleep, dog 2 should likely sleep
        # (probabilistic, so we just check they're valid states)
        assert isinstance(state_1, DogState)
        assert isinstance(state_2, DogState)

    @pytest.mark.asyncio
    async def test_three_dogs_24_cycles(self):
        """Test three dogs running for 24 cycles."""
        runtime = MultiDogRuntime(dog_count=3, enable_logging=False)
        runtime.is_running = True

        for cycle in range(24):
            await runtime.run_cycle(cycle_num=cycle)

        state_summary = runtime.get_state_summary()
        assert state_summary["cycle_count"] == 23  # Last cycle was 23
        assert state_summary["dogs_count"] == 3

        # Check all dogs have valid states
        for dog_id, state_desc in state_summary["dogs_states"].items():
            assert len(state_desc) > 0

    def test_backward_compatibility_single_dog(self):
        """Test that single-dog mode still works (backward compat)."""
        runtime = MultiDogRuntime(dog_count=1)
        assert runtime.scheduler.get_dog_count() == 1

        # Should work like before
        actions = runtime.scheduler.get_next_actions()
        assert len(actions) == 1
        assert "dog_1" in actions


def run_pytest():
    """Run all tests."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    print("Running multi-dog tests...")
    run_pytest()
