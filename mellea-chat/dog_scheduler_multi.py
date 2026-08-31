"""Multi-dog scheduler managing independent state for multiple dogs.

Each dog maintains its own DailySchedule and DogStateMetrics, operating
independently while sharing the same time-of-day reference.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from dog_scheduler import DogState, DogStateMetrics, DailySchedule


@dataclass
class DogInstance:
    """Represents a single dog instance with independent state and behavior."""

    dog_id: str
    scheduler: DailySchedule = field(default_factory=DailySchedule)
    current_state: DogState = field(default=DogState.ALERT)

    def get_next_action(self, dt: Optional[datetime] = None) -> tuple[DogState, int]:
        """Get next behavior for this dog."""
        return self.scheduler.get_next_action(dt)

    def update_state(self, elapsed_ms: int, user_interacted: bool = False):
        """Update this dog's state and metrics."""
        self.scheduler.update_state(elapsed_ms, user_interacted, self.current_state)

    def get_metrics(self) -> DogStateMetrics:
        """Get current metrics for this dog."""
        return self.scheduler.metrics

    def describe_state(self) -> str:
        """Get human-readable state description for this dog."""
        return self.scheduler.describe_current_state()


class MultiDogScheduler:
    """Manages behavior scheduling for multiple independent dogs."""

    def __init__(self, dog_count: int = 2):
        """
        Initialize scheduler for multiple dogs.

        Args:
            dog_count: Number of dogs to manage
        """
        self.dogs: dict[str, DogInstance] = {}
        for i in range(dog_count):
            dog_id = f"dog_{i+1}"
            self.dogs[dog_id] = DogInstance(dog_id=dog_id)

    def add_dog(self, dog_id: str) -> DogInstance:
        """
        Add a new dog to the scheduler.

        Args:
            dog_id: Unique identifier for the dog

        Returns:
            The created DogInstance
        """
        if dog_id in self.dogs:
            raise ValueError(f"Dog {dog_id} already exists")
        dog = DogInstance(dog_id=dog_id)
        self.dogs[dog_id] = dog
        return dog

    def remove_dog(self, dog_id: str) -> bool:
        """
        Remove a dog from the scheduler.

        Args:
            dog_id: ID of dog to remove

        Returns:
            True if removed, False if not found
        """
        if dog_id in self.dogs:
            del self.dogs[dog_id]
            return True
        return False

    def get_next_actions(self, dt: Optional[datetime] = None) -> dict[str, tuple[DogState, int]]:
        """
        Get next action for all dogs.

        Each dog decides independently based on its own metrics and schedule.

        Args:
            dt: Optional datetime reference (uses now() if not provided)

        Returns:
            Mapping of dog_id → (DogState, duration_ms)
        """
        if dt is None:
            dt = datetime.now()

        actions = {}
        for dog_id, dog in self.dogs.items():
            state, duration = dog.get_next_action(dt)
            actions[dog_id] = (state, duration)
        return actions

    def update_all_states(self, elapsed_map: dict[str, int], interactions: Optional[dict[str, bool]] = None):
        """
        Update state for all dogs after their behaviors.

        Args:
            elapsed_map: Mapping of dog_id → elapsed_ms during their behavior
            interactions: Optional mapping of dog_id → user_interacted boolean
        """
        if interactions is None:
            interactions = {}

        for dog_id, dog in self.dogs.items():
            elapsed = elapsed_map.get(dog_id, 0)
            interacted = interactions.get(dog_id, False)
            dog.update_state(elapsed, interacted)

    def get_dog(self, dog_id: str) -> Optional[DogInstance]:
        """Get a specific dog by ID."""
        return self.dogs.get(dog_id)

    def get_all_dogs(self) -> dict[str, DogInstance]:
        """Get all dogs."""
        return dict(self.dogs)

    def get_dog_count(self) -> int:
        """Get number of managed dogs."""
        return len(self.dogs)

    def get_all_metrics(self) -> dict[str, DogStateMetrics]:
        """Get metrics for all dogs."""
        return {dog_id: dog.get_metrics() for dog_id, dog in self.dogs.items()}

    def get_all_states(self) -> dict[str, str]:
        """Get state descriptions for all dogs."""
        return {dog_id: dog.describe_state() for dog_id, dog in self.dogs.items()}

    def describe_all_states(self) -> str:
        """Get multi-line state description for all dogs."""
        lines = []
        for dog_id, dog in self.dogs.items():
            lines.append(f"  {dog.describe_state()}")
        return "\n".join(lines)

    def initiate_interaction(
        self,
        dog_1_id: str,
        dog_2_id: str,
        interaction_type: str,
    ) -> bool:
        """
        Initiate interaction between two dogs.

        Args:
            dog_1_id: First dog ID
            dog_2_id: Second dog ID
            interaction_type: Type of interaction ("chase", "play_together", "interacting")

        Returns:
            True if interaction was started successfully
        """
        dog_1 = self.get_dog(dog_1_id)
        dog_2 = self.get_dog(dog_2_id)

        if not dog_1 or not dog_2:
            return False

        # Determine roles based on interaction type
        if interaction_type == "chase":
            # Higher energy dog chases, lower energy dog flees
            if dog_1.scheduler.metrics.energy_level > dog_2.scheduler.metrics.energy_level:
                pursuer, runner = dog_1, dog_2
            else:
                pursuer, runner = dog_2, dog_1
            pursuer.scheduler.metrics.start_interaction(runner.dog_id, interaction_type)
            runner.scheduler.metrics.start_interaction(pursuer.dog_id, interaction_type)
        else:
            # Mutual interaction
            dog_1.scheduler.metrics.start_interaction(dog_2_id, interaction_type)
            dog_2.scheduler.metrics.start_interaction(dog_1_id, interaction_type)

        return True

    def end_interaction(self, dog_id: str) -> bool:
        """
        End interaction for a dog.

        Args:
            dog_id: Dog ID

        Returns:
            True if interaction was ended
        """
        dog = self.get_dog(dog_id)
        if not dog:
            return False

        dog.scheduler.metrics.end_interaction()
        return True

    def update_interactions(self, elapsed_map: dict[str, int]):
        """
        Update interaction states for all dogs.

        Can end interactions if they've run long enough.

        Args:
            elapsed_map: Mapping of dog_id -> elapsed_ms during behavior
        """
        for dog_id, dog in self.dogs.items():
            metrics = dog.scheduler.metrics

            # If dog is in interaction and interaction energy bonus decayed significantly, end it
            if metrics.interaction_target and metrics.interaction_energy_bonus > 0:
                metrics.interaction_energy_bonus = max(0, metrics.interaction_energy_bonus - 0.05)

                # Probability to end interaction
                if metrics.interaction_energy_bonus <= 0.05 and random.random() < 0.3:
                    self.end_interaction(dog_id)
