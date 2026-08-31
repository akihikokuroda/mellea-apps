"""Multi-dog behavior orchestrator for concurrent dog movement generation.

Maps multiple dogs' states to movement sequences with motor ID prefixing.
Enables multiple dogs to move simultaneously with independent behaviors.
"""

from typing import Optional
from dog_scheduler import DogState, DogStateMetrics
from dog_behavior_orchestrator import BehaviorOrchestrator
from dogmove import MotionFrame
from dog_interaction_movements import (
    chase_sequence,
    play_together_sequence,
    interacting_sequence,
)


class MultiDogOrchestrator:
    """Orchestrates movements for multiple dogs by prefixing motor IDs."""

    def __init__(self, use_variation: bool = True):
        """
        Initialize multi-dog orchestrator.

        Args:
            use_variation: Add randomization to movements for naturalness
        """
        self.behavior_orchestrator = BehaviorOrchestrator(use_variation=use_variation)

    def get_movements_for_all_dogs(
        self,
        dog_behaviors: dict[str, DogState],
        intensity_map: Optional[dict[str, float]] = None,
        duration_map: Optional[dict[str, int]] = None,
        interaction_pairs: Optional[list[tuple[str, str, str]]] = None,
    ) -> list[MotionFrame]:
        """
        Generate movement sequences for multiple dogs with prefixed motor IDs.

        Supports both individual behaviors and coordinated interactions.

        Args:
            dog_behaviors: Mapping of dog_id → DogState (e.g., {"dog_1": DogState.PLAYING})
            intensity_map: Optional mapping of dog_id → intensity (0.0-1.0)
            duration_map: Optional mapping of dog_id → duration_ms
            interaction_pairs: Optional list of (dog_1_id, dog_2_id, interaction_type) tuples

        Returns:
            Merged list of MotionFrame objects with prefixed motor IDs
            (e.g., motor_angles = {"dog_1:tail": 45.0, "dog_2:head": 15.0})
        """
        if not dog_behaviors:
            return []

        if intensity_map is None:
            intensity_map = {dog_id: 0.7 for dog_id in dog_behaviors.keys()}

        if duration_map is None:
            duration_map = {dog_id: 3000 for dog_id in dog_behaviors.keys()}

        # Track which dogs are in interactions
        dogs_in_interaction = set()
        interaction_movements: list[MotionFrame] = []

        # Handle interaction pairs with synchronized movements
        if interaction_pairs:
            for dog_1_id, dog_2_id, interaction_type in interaction_pairs:
                if dog_1_id not in dog_behaviors or dog_2_id not in dog_behaviors:
                    continue

                intensity = min(
                    intensity_map.get(dog_1_id, 0.7),
                    intensity_map.get(dog_2_id, 0.7),
                )
                duration = min(
                    duration_map.get(dog_1_id, 3000),
                    duration_map.get(dog_2_id, 3000),
                )

                # Generate interaction-specific movements
                if interaction_type == "chase":
                    movements = chase_sequence(dog_1_id, dog_2_id, duration, intensity)
                elif interaction_type == "play_together":
                    movements = play_together_sequence(dog_1_id, dog_2_id, duration, intensity)
                else:  # "interacting" or generic
                    movements = interacting_sequence(dog_1_id, dog_2_id, duration, interaction_type)

                interaction_movements.extend(movements)
                dogs_in_interaction.add(dog_1_id)
                dogs_in_interaction.add(dog_2_id)

        # Collect movements for each dog not in interactions
        all_movements: list[list[MotionFrame]] = []
        for dog_id, dog_state in dog_behaviors.items():
            if dog_id in dogs_in_interaction:
                continue

            intensity = intensity_map.get(dog_id, 0.7)
            duration = duration_map.get(dog_id, 3000)

            # Get movements from single-dog orchestrator
            movements = self.behavior_orchestrator.get_movements_for_state(
                dog_state=dog_state,
                intensity=intensity,
                duration=duration,
            )

            # Prefix motor IDs with dog_id
            prefixed_movements = self._prefix_movements(movements, dog_id)
            all_movements.append(prefixed_movements)

        # Add interaction movements to collection
        if interaction_movements:
            all_movements.append(interaction_movements)

        # Merge all movements into a single timeline
        merged = self._merge_movements(all_movements)
        return merged

    def _prefix_movements(self, movements: list[MotionFrame], dog_id: str) -> list[MotionFrame]:
        """
        Prefix all motor IDs in movement frames with dog_id.

        Example: "tail" → "dog_1:tail"

        Args:
            movements: Original movement frames
            dog_id: Dog identifier (e.g., "dog_1")

        Returns:
            Movement frames with prefixed motor IDs
        """
        prefixed = []
        for frame in movements:
            prefixed_angles = {
                f"{dog_id}:{motor_id}": angle
                for motor_id, angle in frame.motor_angles.items()
            }
            prefixed.append(MotionFrame(
                timestamp_ms=frame.timestamp_ms,
                motor_angles=prefixed_angles,
            ))
        return prefixed

    def _merge_movements(self, all_movements: list[list[MotionFrame]]) -> list[MotionFrame]:
        """
        Merge multiple dogs' movements into a single timeline.

        Combines motor_angles dicts from different dogs at each timestamp.
        If timestamps don't align, interpolates to unified timeline.

        Args:
            all_movements: List of (dog's movements list) for each dog

        Returns:
            Merged MotionFrame list with all dogs' motors
        """
        if not all_movements:
            return []

        # Collect all unique timestamps
        all_timestamps = set()
        for dog_movements in all_movements:
            for frame in dog_movements:
                all_timestamps.add(frame.timestamp_ms)

        if not all_timestamps:
            return []

        # Sort timestamps
        sorted_timestamps = sorted(all_timestamps)

        # Build merged frames
        merged_frames = []
        for timestamp in sorted_timestamps:
            merged_angles = {}

            # Collect angles from each dog at this timestamp
            for dog_movements in all_movements:
                # Find frame at or closest to this timestamp
                frame_at_ts = self._get_frame_at_timestamp(dog_movements, timestamp)
                if frame_at_ts:
                    merged_angles.update(frame_at_ts.motor_angles)

            if merged_angles:
                merged_frames.append(MotionFrame(
                    timestamp_ms=timestamp,
                    motor_angles=merged_angles,
                ))

        return merged_frames

    def _get_frame_at_timestamp(
        self,
        movements: list[MotionFrame],
        target_timestamp: float,
    ) -> Optional[MotionFrame]:
        """
        Get the frame at or closest to a target timestamp.

        Args:
            movements: List of movement frames
            target_timestamp: Target timestamp in milliseconds

        Returns:
            Frame at exact timestamp or closest frame before it, or None
        """
        if not movements:
            return None

        # Find exact match first
        for frame in movements:
            if frame.timestamp_ms == target_timestamp:
                return frame

        # Find closest frame before target
        closest = None
        for frame in movements:
            if frame.timestamp_ms <= target_timestamp:
                if closest is None or frame.timestamp_ms > closest.timestamp_ms:
                    closest = frame

        return closest
