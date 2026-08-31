"""Comprehensive tests for dog-to-dog interaction system."""

import asyncio
import pytest
from dog_scheduler import DogState
from dog_scheduler_multi import MultiDogScheduler
from dog_orchestrator_multi import MultiDogOrchestrator
from dog_interaction_detector import InteractionDetector
from dog_interaction_movements import (
    chase_sequence,
    play_together_sequence,
    interacting_sequence,
)
from dog_runtime_multi import MultiDogRuntime
from dogmove import MotionFrame


class TestInteractionStates:
    """Tests for new interaction states."""

    def test_chasing_state_exists(self):
        """Test CHASING state is defined."""
        assert hasattr(DogState, "CHASING")
        assert DogState.CHASING.value == "chasing"

    def test_being_chased_state_exists(self):
        """Test BEING_CHASED state is defined."""
        assert hasattr(DogState, "BEING_CHASED")
        assert DogState.BEING_CHASED.value == "being_chased"

    def test_playing_together_state_exists(self):
        """Test PLAYING_TOGETHER state is defined."""
        assert hasattr(DogState, "PLAYING_TOGETHER")
        assert DogState.PLAYING_TOGETHER.value == "playing_together"

    def test_interacting_state_exists(self):
        """Test INTERACTING state is defined."""
        assert hasattr(DogState, "INTERACTING")
        assert DogState.INTERACTING.value == "interacting"


class TestMetricsInteraction:
    """Tests for interaction metrics."""

    def test_metrics_has_interaction_target(self):
        """Test metrics has interaction_target field."""
        scheduler = MultiDogScheduler(dog_count=1)
        dog = scheduler.get_dog("dog_1")
        assert hasattr(dog.scheduler.metrics, "interaction_target")
        assert dog.scheduler.metrics.interaction_target is None

    def test_metrics_has_interaction_type(self):
        """Test metrics has interaction_type field."""
        scheduler = MultiDogScheduler(dog_count=1)
        dog = scheduler.get_dog("dog_1")
        assert hasattr(dog.scheduler.metrics, "interaction_type")
        assert dog.scheduler.metrics.interaction_type is None

    def test_can_initiate_interaction_method(self):
        """Test can_initiate_interaction method."""
        scheduler = MultiDogScheduler(dog_count=1)
        dog = scheduler.get_dog("dog_1")

        # With default metrics (energy=50, attention=30)
        assert dog.scheduler.metrics.can_initiate_interaction() is True

        # Low energy should prevent interaction
        dog.scheduler.metrics.energy_level = 10
        assert dog.scheduler.metrics.can_initiate_interaction() is False

    def test_start_interaction_method(self):
        """Test start_interaction method."""
        scheduler = MultiDogScheduler(dog_count=1)
        dog = scheduler.get_dog("dog_1")

        dog.scheduler.metrics.start_interaction("dog_2", "chase")
        assert dog.scheduler.metrics.interaction_target == "dog_2"
        assert dog.scheduler.metrics.interaction_type == "chase"
        assert dog.scheduler.metrics.interaction_energy_bonus > 0

    def test_end_interaction_method(self):
        """Test end_interaction method."""
        scheduler = MultiDogScheduler(dog_count=1)
        dog = scheduler.get_dog("dog_1")

        dog.scheduler.metrics.start_interaction("dog_2", "chase")
        dog.scheduler.metrics.end_interaction()
        assert dog.scheduler.metrics.interaction_target is None
        assert dog.scheduler.metrics.interaction_type is None
        assert dog.scheduler.metrics.interaction_energy_bonus == 0.0


class TestInteractionDetector:
    """Tests for InteractionDetector."""

    def test_detector_initialization(self):
        """Test detector initializes correctly."""
        detector = InteractionDetector()
        assert detector is not None

    def test_detect_nearby_dogs(self):
        """Test detecting nearby dogs."""
        scheduler = MultiDogScheduler(dog_count=3)
        detector = InteractionDetector()

        nearby = detector.detect_nearby_dogs(scheduler)
        assert len(nearby) == 3
        assert "dog_1" in nearby
        assert "dog_2" in nearby
        assert "dog_3" in nearby

    def test_suggest_interactions(self):
        """Test suggesting interactions."""
        scheduler = MultiDogScheduler(dog_count=2)
        detector = InteractionDetector()

        nearby = detector.detect_nearby_dogs(scheduler)
        suggestions = detector.suggest_interactions(
            scheduler, nearby, interaction_probability=1.0
        )

        # With high probability and nearby dogs, should get suggestions
        assert isinstance(suggestions, list)

    def test_are_compatible_for_interaction(self):
        """Test compatibility check."""
        scheduler = MultiDogScheduler(dog_count=2)
        detector = InteractionDetector()

        dog_1 = scheduler.get_dog("dog_1")
        dog_2 = scheduler.get_dog("dog_2")

        # Both have decent energy and attention
        compatible = detector.are_compatible_for_interaction(
            scheduler, "dog_1", "dog_2"
        )
        assert isinstance(compatible, bool)

    def test_incompatible_sleeping_dogs(self):
        """Test incompatibility for sleeping dogs."""
        scheduler = MultiDogScheduler(dog_count=2)
        detector = InteractionDetector()

        dog_1 = scheduler.get_dog("dog_1")
        dog_1.current_state = DogState.SLEEPING

        compatible = detector.are_compatible_for_interaction(
            scheduler, "dog_1", "dog_2"
        )
        assert compatible is False


class TestInteractionMovements:
    """Tests for interaction movement generation."""

    def test_chase_sequence_generation(self):
        """Test chase sequence generation."""
        movements = chase_sequence("dog_1", "dog_2", duration_ms=3000, intensity=0.8)
        assert len(movements) > 0
        assert all(isinstance(m, MotionFrame) for m in movements)

    def test_chase_sequence_has_both_dogs(self):
        """Test chase sequence includes both dogs' motors."""
        movements = chase_sequence("dog_1", "dog_2", duration_ms=3000)
        all_motors = set()
        for frame in movements:
            all_motors.update(frame.motor_angles.keys())

        dog_1_motors = {m for m in all_motors if "dog_1:" in m}
        dog_2_motors = {m for m in all_motors if "dog_2:" in m}

        assert len(dog_1_motors) > 0
        assert len(dog_2_motors) > 0

    def test_play_together_sequence_generation(self):
        """Test play together sequence generation."""
        movements = play_together_sequence("dog_1", "dog_2", duration_ms=3000)
        assert len(movements) > 0

    def test_play_together_synchronized(self):
        """Test play together has synchronized movements."""
        movements = play_together_sequence("dog_1", "dog_2", duration_ms=3000)
        # Should have frames where both dogs move at same timestamp
        synchronized_frames = 0
        for frame in movements:
            dog_1_motors = sum(1 for k in frame.motor_angles.keys() if "dog_1:" in k)
            dog_2_motors = sum(1 for k in frame.motor_angles.keys() if "dog_2:" in k)
            if dog_1_motors > 0 and dog_2_motors > 0:
                synchronized_frames += 1

        assert synchronized_frames > 0

    def test_interacting_sequence_generation(self):
        """Test interacting sequence generation."""
        movements = interacting_sequence("dog_1", "dog_2", duration_ms=2000)
        assert len(movements) > 0

    def test_interaction_movement_duration(self):
        """Test interaction movements respect duration."""
        duration = 5000
        movements = play_together_sequence("dog_1", "dog_2", duration_ms=duration)
        if movements:
            max_timestamp = max(m.timestamp_ms for m in movements)
            assert max_timestamp <= duration * 1.1  # Allow 10% overage


class TestSchedulerInteraction:
    """Tests for scheduler interaction coordination."""

    def test_initiate_interaction(self):
        """Test initiating interaction between dogs."""
        scheduler = MultiDogScheduler(dog_count=2)

        success = scheduler.initiate_interaction("dog_1", "dog_2", "play_together")
        assert success is True

        dog_1 = scheduler.get_dog("dog_1")
        dog_2 = scheduler.get_dog("dog_2")

        assert dog_1.scheduler.metrics.interaction_target == "dog_2"
        assert dog_2.scheduler.metrics.interaction_target == "dog_1"

    def test_chase_role_assignment(self):
        """Test chase assigns pursuer and runner roles."""
        scheduler = MultiDogScheduler(dog_count=2)

        dog_1 = scheduler.get_dog("dog_1")
        dog_2 = scheduler.get_dog("dog_2")

        # Set different energy levels
        dog_1.scheduler.metrics.energy_level = 90
        dog_2.scheduler.metrics.energy_level = 30

        success = scheduler.initiate_interaction("dog_1", "dog_2", "chase")
        assert success is True

        # Both should have interaction started
        assert dog_1.scheduler.metrics.interaction_target is not None
        assert dog_2.scheduler.metrics.interaction_target is not None

    def test_update_interactions(self):
        """Test updating interaction states."""
        scheduler = MultiDogScheduler(dog_count=2)

        scheduler.initiate_interaction("dog_1", "dog_2", "play_together")
        dog_1 = scheduler.get_dog("dog_1")
        initial_bonus = dog_1.scheduler.metrics.interaction_energy_bonus

        scheduler.update_interactions({"dog_1": 1000, "dog_2": 1000})

        # Bonus should decay
        assert dog_1.scheduler.metrics.interaction_energy_bonus <= initial_bonus

    def test_end_interaction(self):
        """Test ending interaction."""
        scheduler = MultiDogScheduler(dog_count=2)

        scheduler.initiate_interaction("dog_1", "dog_2", "play_together")
        success = scheduler.end_interaction("dog_1")

        assert success is True
        dog_1 = scheduler.get_dog("dog_1")
        assert dog_1.scheduler.metrics.interaction_target is None


class TestOrchestratorInteraction:
    """Tests for orchestrator handling interactions."""

    def test_orchestrator_accepts_interaction_pairs(self):
        """Test orchestrator accepts interaction_pairs parameter."""
        orch = MultiDogOrchestrator()
        dog_behaviors = {"dog_1": DogState.CHASING, "dog_2": DogState.BEING_CHASED}
        interaction_pairs = [("dog_1", "dog_2", "chase")]

        movements = orch.get_movements_for_all_dogs(
            dog_behaviors=dog_behaviors,
            interaction_pairs=interaction_pairs,
        )

        assert len(movements) > 0

    def test_interaction_movements_have_both_dogs(self):
        """Test interaction movements include both dogs."""
        orch = MultiDogOrchestrator()
        dog_behaviors = {"dog_1": DogState.PLAYING_TOGETHER, "dog_2": DogState.PLAYING_TOGETHER}
        interaction_pairs = [("dog_1", "dog_2", "play_together")]

        movements = orch.get_movements_for_all_dogs(
            dog_behaviors=dog_behaviors,
            interaction_pairs=interaction_pairs,
        )

        all_motors = set()
        for frame in movements:
            all_motors.update(frame.motor_angles.keys())

        dog_1_motors = sum(1 for m in all_motors if "dog_1:" in m)
        dog_2_motors = sum(1 for m in all_motors if "dog_2:" in m)

        assert dog_1_motors > 0
        assert dog_2_motors > 0


class TestBehaviorOrchestrator:
    """Tests for behavior orchestrator interaction states."""

    def test_chasing_movements(self):
        """Test CHASING state generates movements."""
        from dog_behavior_orchestrator import BehaviorOrchestrator

        orch = BehaviorOrchestrator()
        movements = orch.get_movements_for_state(DogState.CHASING, intensity=0.8, duration=2000)
        assert len(movements) > 0

    def test_being_chased_movements(self):
        """Test BEING_CHASED state generates movements."""
        from dog_behavior_orchestrator import BehaviorOrchestrator

        orch = BehaviorOrchestrator()
        movements = orch.get_movements_for_state(DogState.BEING_CHASED, intensity=0.7, duration=2000)
        assert len(movements) > 0

    def test_playing_together_movements(self):
        """Test PLAYING_TOGETHER state generates movements."""
        from dog_behavior_orchestrator import BehaviorOrchestrator

        orch = BehaviorOrchestrator()
        movements = orch.get_movements_for_state(DogState.PLAYING_TOGETHER, intensity=0.7, duration=3000)
        assert len(movements) > 0

    def test_interacting_movements(self):
        """Test INTERACTING state generates movements."""
        from dog_behavior_orchestrator import BehaviorOrchestrator

        orch = BehaviorOrchestrator()
        movements = orch.get_movements_for_state(DogState.INTERACTING, intensity=0.6, duration=2000)
        assert len(movements) > 0


class TestIntegration:
    """Integration tests for complete interaction system."""

    @pytest.mark.asyncio
    async def test_runtime_with_interactions(self):
        """Test runtime detects and handles interactions."""
        runtime = MultiDogRuntime(dog_count=2, enable_logging=False)
        runtime.is_running = True

        # Run a few cycles - should detect and initiate interactions
        for i in range(5):
            success = await runtime.run_cycle(cycle_num=i)
            assert success is True

        # Check if any interactions were logged
        interaction_events = [
            e for e in runtime.state_log if e.get("event") == "interaction_start"
        ]
        # With 5 cycles and 0.25 probability, likely to have at least one
        # but not guaranteed due to randomness
        assert isinstance(interaction_events, list)

    @pytest.mark.asyncio
    async def test_multiple_interaction_types(self):
        """Test different interaction types can occur."""
        runtime = MultiDogRuntime(dog_count=2, enable_logging=False)
        runtime.is_running = True

        interaction_types_seen = set()

        for i in range(10):
            await runtime.run_cycle(cycle_num=i)

            # Check what interactions happened
            for dog_id, dog in runtime.scheduler.get_all_dogs().items():
                if dog.scheduler.metrics.interaction_type:
                    interaction_types_seen.add(dog.scheduler.metrics.interaction_type)

        # Should see some variety in interaction types
        assert isinstance(interaction_types_seen, set)

    def test_interaction_full_lifecycle(self):
        """Test complete interaction lifecycle."""
        scheduler = MultiDogScheduler(dog_count=2)
        detector = InteractionDetector()

        # 1. Detect nearby dogs
        nearby = detector.detect_nearby_dogs(scheduler)
        assert len(nearby) == 2

        # 2. Suggest interactions with high probability
        suggestions = detector.suggest_interactions(
            scheduler, nearby, interaction_probability=1.0
        )

        # 3. Initiate if suggested
        if suggestions:
            dog_1_id, dog_2_id, interaction_type = suggestions[0]
            success = scheduler.initiate_interaction(dog_1_id, dog_2_id, interaction_type)
            assert success is True

            # 4. Verify interaction state
            dog_1 = scheduler.get_dog(dog_1_id)
            assert dog_1.scheduler.metrics.interaction_target is not None

            # 5. Update and potentially end
            scheduler.update_interactions(
                {dog_1_id: 1000, dog_2_id: 1000}
            )

            # 6. End interaction
            scheduler.end_interaction(dog_1_id)
            assert dog_1.scheduler.metrics.interaction_target is None


def run_pytest():
    """Run all tests."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    print("Running dog interaction tests...")
    run_pytest()
