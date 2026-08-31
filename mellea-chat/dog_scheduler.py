from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random
from typing import Optional


class DogState(Enum):
    SLEEPING = "sleeping"
    ALERT = "alert"
    PLAYING = "playing"
    RESTING = "resting"
    EXPLORING = "exploring"
    EATING = "eating"
    SCRATCHING = "scratching"
    GROOMING = "grooming"
    CHASING = "chasing"
    BEING_CHASED = "being_chased"
    PLAYING_TOGETHER = "playing_together"
    INTERACTING = "interacting"


@dataclass
class DogStateMetrics:
    energy_level: int = 50  # 0-100
    hunger_level: int = 50  # 0-100
    attention_level: int = 30  # 0-100, how much dog wants interaction
    interaction_target: Optional[str] = None  # ID of dog this dog is interacting with
    interaction_type: Optional[str] = None  # Type of interaction: "chase", "play_together", etc.
    interaction_energy_bonus: float = 0.0  # Extra energy during interactions (0.0-1.0)

    def decay_over_time(self, elapsed_ms: int):
        """Simulate natural decay of metrics over time."""
        elapsed_min = elapsed_ms / 60000

        # Energy decays from activity, recovers from rest/sleep
        self.energy_level = max(0, self.energy_level - int(elapsed_min * 0.5))

        # Hunger increases over time
        self.hunger_level = min(100, self.hunger_level + int(elapsed_min * 0.3))

        # Attention naturally decays
        self.attention_level = max(0, self.attention_level - int(elapsed_min * 0.2))

    def recover_from_sleep(self, elapsed_ms: int):
        """Recovery during sleep."""
        elapsed_min = elapsed_ms / 60000
        self.energy_level = min(100, self.energy_level + int(elapsed_min * 1.5))
        self.hunger_level = min(100, self.hunger_level + int(elapsed_min * 0.1))

    def recover_from_rest(self, elapsed_ms: int):
        """Recovery during rest."""
        elapsed_min = elapsed_ms / 60000
        self.energy_level = min(100, self.energy_level + int(elapsed_min * 0.8))

    def after_eating(self):
        """Reset hunger after eating."""
        self.hunger_level = 0

    def on_user_interaction(self, intensity: int = 50):
        """Dog responds to user interaction."""
        self.attention_level = min(100, intensity)

    def can_initiate_interaction(self) -> bool:
        """Check if dog has energy and attention for interaction."""
        return self.energy_level > 40 and self.attention_level > 25 and self.interaction_target is None

    def start_interaction(self, other_dog_id: str, interaction_type: str):
        """Start interaction with another dog."""
        self.interaction_target = other_dog_id
        self.interaction_type = interaction_type
        self.interaction_energy_bonus = 0.2

    def end_interaction(self):
        """End current interaction."""
        self.interaction_target = None
        self.interaction_type = None
        self.interaction_energy_bonus = 0.0


@dataclass
class DailySchedule:
    """Manages dog's 24-hour lifecycle with realistic behavior patterns."""

    # Base activity for each hour (0-23)
    hourly_base_activity: dict[int, DogState] = field(default_factory=lambda: {
        0: DogState.SLEEPING,      # 00:00 - deep sleep
        1: DogState.SLEEPING,      # 01:00
        2: DogState.SLEEPING,      # 02:00
        3: DogState.SLEEPING,      # 03:00
        4: DogState.SLEEPING,      # 04:00
        5: DogState.SLEEPING,      # 05:00 - early morning
        6: DogState.ALERT,         # 06:00 - waking up
        7: DogState.ALERT,         # 07:00 - morning routine
        8: DogState.GROOMING,      # 08:00 - grooming
        9: DogState.EXPLORING,     # 09:00 - morning exploration
        10: DogState.PLAYING,      # 10:00 - play time
        11: DogState.PLAYING,      # 11:00
        12: DogState.EATING,       # 12:00 - lunch
        13: DogState.RESTING,      # 13:00 - post-lunch rest
        14: DogState.RESTING,      # 14:00
        15: DogState.EXPLORING,    # 15:00 - afternoon exploration
        16: DogState.PLAYING,      # 16:00 - afternoon play
        17: DogState.PLAYING,      # 17:00
        18: DogState.EATING,       # 18:00 - dinner
        19: DogState.RESTING,      # 19:00 - evening rest
        20: DogState.ALERT,        # 20:00 - evening alert
        21: DogState.GROOMING,     # 21:00 - grooming/wind-down
        22: DogState.SLEEPING,     # 22:00 - sleep prep
        23: DogState.SLEEPING,     # 23:00 - sleeping
    })

    metrics: DogStateMetrics = field(default_factory=DogStateMetrics)
    current_state: DogState = DogState.ALERT
    state_start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)

    def get_base_activity_for_time(self, dt: Optional[datetime] = None) -> DogState:
        """Get the base activity from the schedule for a given hour."""
        if dt is None:
            dt = datetime.now()
        hour = dt.hour
        return self.hourly_base_activity.get(hour, DogState.ALERT)

    def get_next_action(self, dt: Optional[datetime] = None) -> tuple[DogState, int]:
        """
        Determine next action based on:
        - Time of day (schedule)
        - Internal metrics (energy, hunger, attention)
        - Active interactions
        - Some randomization for naturalness

        Returns: (DogState, duration_ms)
        """
        if dt is None:
            dt = datetime.now()

        # If currently in an interaction, continue with appropriate state
        if self.metrics.interaction_target and self.metrics.interaction_type:
            if self.metrics.interaction_type == "chase":
                next_state = random.choice([DogState.CHASING, DogState.PLAYING])
            elif self.metrics.interaction_type == "play_together":
                next_state = DogState.PLAYING_TOGETHER
            else:
                next_state = DogState.INTERACTING
            duration = random.randint(5000, 20000)  # 5-20 sec per interaction cycle
            return next_state, duration

        base_state = self.get_base_activity_for_time(dt)

        # Override schedule based on metrics
        if self.metrics.energy_level < 20:
            next_state = DogState.SLEEPING
            duration = random.randint(600000, 1800000)  # 10-30 min
            return next_state, duration

        if self.metrics.hunger_level > 80:
            next_state = DogState.EATING
            duration = random.randint(5000, 15000)  # 5-15 sec
            return next_state, duration

        if self.metrics.energy_level > 80 and base_state != DogState.EATING:
            # Dog is very energetic
            if self.metrics.attention_level > 40:
                next_state = DogState.PLAYING
            else:
                next_state = DogState.EXPLORING
            duration = random.randint(10000, 30000)  # 10-30 sec
            return next_state, duration

        # Add some randomness to base state
        if random.random() < 0.15:  # 15% chance of deviation
            alternatives = [s for s in DogState if s != base_state and s != DogState.SLEEPING]
            if alternatives:
                next_state = random.choice(alternatives)
            else:
                next_state = base_state
        else:
            next_state = base_state

        # Determine duration based on state
        duration_map = {
            DogState.SLEEPING: random.randint(600000, 3600000),    # 10-60 min
            DogState.ALERT: random.randint(5000, 20000),           # 5-20 sec
            DogState.PLAYING: random.randint(15000, 45000),        # 15-45 sec
            DogState.RESTING: random.randint(30000, 120000),       # 30-120 sec
            DogState.EXPLORING: random.randint(20000, 60000),      # 20-60 sec
            DogState.EATING: random.randint(5000, 15000),          # 5-15 sec
            DogState.SCRATCHING: random.randint(3000, 10000),      # 3-10 sec
            DogState.GROOMING: random.randint(10000, 30000),       # 10-30 sec
            DogState.CHASING: random.randint(10000, 30000),        # 10-30 sec
            DogState.BEING_CHASED: random.randint(10000, 30000),   # 10-30 sec
            DogState.PLAYING_TOGETHER: random.randint(15000, 45000),  # 15-45 sec
            DogState.INTERACTING: random.randint(5000, 20000),     # 5-20 sec
        }
        duration = duration_map.get(next_state, 10000)

        return next_state, duration

    def update_state(self, elapsed_ms: int, user_interacted: bool = False, new_state: Optional[DogState] = None):
        """Update internal state after behavior execution."""
        now = datetime.now()
        time_delta = now - self.last_update
        actual_elapsed_ms = int(time_delta.total_seconds() * 1000)

        # Update current state if provided
        if new_state:
            self.current_state = new_state
            self.state_start_time = now

        # Update metrics based on current state
        if self.current_state == DogState.SLEEPING:
            self.metrics.recover_from_sleep(actual_elapsed_ms)
        elif self.current_state == DogState.RESTING:
            self.metrics.recover_from_rest(actual_elapsed_ms)
        elif self.current_state == DogState.EATING:
            self.metrics.after_eating()
        elif self.current_state == DogState.PLAYING or self.current_state == DogState.EXPLORING:
            self.metrics.energy_level = max(0, self.metrics.energy_level - int(actual_elapsed_ms / 60000))

        # General decay
        self.metrics.decay_over_time(actual_elapsed_ms)

        # User interaction
        if user_interacted:
            self.metrics.on_user_interaction(intensity=70)

        # Clamp values
        self.metrics.energy_level = max(0, min(100, self.metrics.energy_level))
        self.metrics.hunger_level = max(0, min(100, self.metrics.hunger_level))
        self.metrics.attention_level = max(0, min(100, self.metrics.attention_level))

        self.last_update = now

    def describe_current_state(self) -> str:
        """Human-readable description of dog's current state."""
        state_name = self.current_state.value
        energy_desc = "energetic" if self.metrics.energy_level > 70 else "tired" if self.metrics.energy_level < 30 else "normal"
        hunger_desc = "hungry" if self.metrics.hunger_level > 70 else "not hungry"

        return f"Dog is {state_name}, feeling {energy_desc} and {hunger_desc}. Energy: {self.metrics.energy_level}%, Hunger: {self.metrics.hunger_level}%, Attention: {self.metrics.attention_level}%"

    def get_metrics_dict(self) -> dict:
        """Return metrics as a dictionary for serialization."""
        return {
            "energy_level": self.metrics.energy_level,
            "hunger_level": self.metrics.hunger_level,
            "attention_level": self.metrics.attention_level,
            "current_state": self.current_state.value,
            "interaction_target": self.metrics.interaction_target,
            "interaction_type": self.metrics.interaction_type,
            "timestamp": self.last_update.isoformat(),
        }
