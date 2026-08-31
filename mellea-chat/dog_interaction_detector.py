"""Detects and suggests dog-to-dog interactions."""

import random
from typing import Optional
from dog_scheduler import DogState
from dog_scheduler_multi import MultiDogScheduler, DogInstance


class InteractionDetector:
    """Detects nearby dogs and suggests compatible interactions."""

    def detect_nearby_dogs(
        self,
        scheduler: MultiDogScheduler,
        proximity_threshold: int = 50,
    ) -> dict[str, list[str]]:
        """
        Detect which dogs are near each other.

        Args:
            scheduler: MultiDogScheduler instance
            proximity_threshold: Virtual proximity threshold (currently randomized)

        Returns:
            Dict mapping dog_id -> list of nearby dog_ids
        """
        nearby = {dog_id: [] for dog_id in scheduler.get_all_dogs().keys()}

        dogs = list(scheduler.get_all_dogs().items())
        for i, (dog_id_1, dog_1) in enumerate(dogs):
            for dog_id_2, dog_2 in dogs[i + 1 :]:
                # Virtual proximity: randomly detect if dogs should be considered "near"
                # In a real system, this would use actual position tracking
                if random.random() < 0.4:  # 40% chance dogs are considered nearby
                    nearby[dog_id_1].append(dog_id_2)
                    nearby[dog_id_2].append(dog_id_1)

        return nearby

    def suggest_interactions(
        self,
        scheduler: MultiDogScheduler,
        nearby_dogs: dict[str, list[str]],
        interaction_probability: float = 0.3,
    ) -> list[tuple[str, str, str]]:
        """
        Suggest compatible interactions between nearby dogs.

        Args:
            scheduler: MultiDogScheduler instance
            nearby_dogs: Dict of dog_id -> nearby dog_ids
            interaction_probability: Probability of suggesting interaction (0.0-1.0)

        Returns:
            List of (dog_1_id, dog_2_id, interaction_type) tuples
        """
        suggestions = []

        for dog_id_1, nearby_list in nearby_dogs.items():
            for dog_id_2 in nearby_list:
                # Avoid duplicate suggestions
                if dog_id_1 > dog_id_2:
                    continue

                # Check if interaction should be suggested
                if random.random() > interaction_probability:
                    continue

                # Check compatibility
                if not self.are_compatible_for_interaction(scheduler, dog_id_1, dog_id_2):
                    continue

                # Determine interaction type
                interaction_type = self._choose_interaction_type(scheduler, dog_id_1, dog_id_2)
                if interaction_type:
                    suggestions.append((dog_id_1, dog_id_2, interaction_type))

        return suggestions

    def are_compatible_for_interaction(
        self,
        scheduler: MultiDogScheduler,
        dog_id_1: str,
        dog_id_2: str,
    ) -> bool:
        """
        Check if two dogs can interact.

        Args:
            scheduler: MultiDogScheduler instance
            dog_id_1: First dog ID
            dog_id_2: Second dog ID

        Returns:
            True if dogs are compatible for interaction
        """
        dog_1 = scheduler.get_dog(dog_id_1)
        dog_2 = scheduler.get_dog(dog_id_2)

        if not dog_1 or not dog_2:
            return False

        # Both dogs need energy and attention
        if not dog_1.scheduler.metrics.can_initiate_interaction():
            return False
        if not dog_2.scheduler.metrics.can_initiate_interaction():
            return False

        # Avoid sleep and eating states
        excluded_states = {DogState.SLEEPING, DogState.EATING}
        if dog_1.current_state in excluded_states or dog_2.current_state in excluded_states:
            return False

        return True

    def _choose_interaction_type(
        self,
        scheduler: MultiDogScheduler,
        dog_id_1: str,
        dog_id_2: str,
    ) -> Optional[str]:
        """
        Determine the type of interaction between dogs.

        Args:
            scheduler: MultiDogScheduler instance
            dog_id_1: First dog ID
            dog_id_2: Second dog ID

        Returns:
            Interaction type: "chase", "play_together", "interacting", or None
        """
        dog_1 = scheduler.get_dog(dog_id_1)
        dog_2 = scheduler.get_dog(dog_id_2)

        if not dog_1 or not dog_2:
            return None

        energy_1 = dog_1.scheduler.metrics.energy_level
        energy_2 = dog_2.scheduler.metrics.energy_level

        # High energy dogs prefer chase or play_together
        if energy_1 > 60 and energy_2 > 60:
            return random.choice(["chase", "play_together"])
        elif energy_1 > 60 or energy_2 > 60:
            # One energetic dog may chase, other may flee
            return "chase"
        else:
            # Lower energy: gentle interaction
            return "interacting"
