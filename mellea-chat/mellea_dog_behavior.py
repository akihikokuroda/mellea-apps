"""LLM-powered behavior advisor for the 24/7 AI dog system."""

import json
from typing import Optional, Tuple
from dog_scheduler import DogState, DogStateMetrics, DailySchedule
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.components import CBlock
import mellea.stdlib.functional as mfuncs


class DogBehaviorLLM:
    """Uses LLM to make intelligent behavior decisions."""

    def __init__(
        self,
        model_id: str = "granite4.1:3b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
    ):
        """
        Initialize the LLM behavior advisor.

        Args:
            model_id: LLM model to use (Ollama)
            base_url: Ollama server URL
            temperature: Sampling temperature (0.3-0.95)
        """
        self.backend = OllamaModelBackend(model_id=model_id, base_url=base_url)
        self.temperature = temperature
        self.model_id = model_id

    async def decide_behavior(
        self, current_state: DogState, metrics: DogStateMetrics, time_of_day: int
    ) -> Tuple[DogState, int]:
        """
        Use LLM to decide next behavior.

        Args:
            current_state: Current dog state
            metrics: Dog's metrics (energy, hunger, attention)
            time_of_day: Hour of day (0-23)

        Returns:
            (next_state, duration_ms)
        """
        # Build context description
        energy_desc = "high" if metrics.energy_level > 70 else "low" if metrics.energy_level < 30 else "normal"
        hunger_desc = "hungry" if metrics.hunger_level > 70 else "not hungry"
        attention_desc = "seeking attention" if metrics.attention_level > 60 else "focused on own activity"

        time_desc = self._describe_time_of_day(time_of_day)

        # Build prompt for LLM
        prompt = f"""You are an AI controlling a toy dog's behavior in real-time.

CURRENT STATUS:
- Time: {time_of_day:02d}:00 ({time_desc})
- Current behavior: {current_state.value}
- Energy level: {metrics.energy_level}% ({energy_desc})
- Hunger level: {metrics.hunger_level}% ({hunger_desc})
- Attention level: {metrics.attention_level}% ({attention_desc})

AVAILABLE BEHAVIORS:
- SLEEPING: Deep rest (duration: 300-3600ms)
- ALERT: Awake and watching (duration: 5-20000ms)
- PLAYING: Active play with high energy (duration: 10-45000ms)
- RESTING: Relaxed rest (duration: 30-120000ms)
- EXPLORING: Curious exploration (duration: 15-60000ms)
- EATING: Feeding time (duration: 5-15000ms)
- SCRATCHING: Self-grooming (duration: 3-10000ms)
- GROOMING: Cleaning routine (duration: 10-30000ms)

Based on the current state and time of day, what should the dog do next?

RESPOND WITH ONLY THIS JSON FORMAT (no extra text):
{{
  "behavior": "PLAYING",
  "duration_ms": 25000,
  "reasoning": "High energy at afternoon playtime"
}}

Behaviors must match the list exactly. Durations must be within the specified ranges."""

        try:
            # Execute LLM call
            ctx = SimpleContext()
            model_options = {
                "temperature": self.temperature,
                "system_prompt": "You are a helpful AI assistant. Always respond with valid JSON only.",
            }
            action = CBlock(prompt)

            mot, gen_ctx = await mfuncs.aact(
                action, ctx, self.backend, strategy=None, model_options=model_options
            )

            # Get response
            response_text = await mot.avalue()

            # Parse JSON response
            result = self._parse_behavior_response(response_text)
            if result:
                next_state, duration = result
                return next_state, duration
        except Exception as e:
            print(f"LLM advisor error: {e}")

        # Fallback: return current state with random duration
        import random
        return current_state, random.randint(5000, 15000)

    async def interpret_user_command(self, command: str, current_state: DogState) -> Optional[DogState]:
        """
        Interpret natural language user commands.

        Args:
            command: User command text (e.g., "make the dog play")
            current_state: Current dog state

        Returns:
            Suggested DogState or None
        """
        prompt = f"""You are interpreting user commands for a toy dog.

COMMAND: "{command}"
CURRENT DOG STATE: {current_state.value}

AVAILABLE BEHAVIORS:
- SLEEPING: dog sleeps/naps
- ALERT: dog is awake and aware
- PLAYING: dog plays actively
- RESTING: dog relaxes
- EXPLORING: dog explores/sniffs around
- EATING: dog eats
- SCRATCHING: dog scratches/grooms
- GROOMING: dog cleans/grooms

What behavior should the dog do based on this command? If the command doesn't match any behavior, respond "NONE".

RESPOND WITH ONLY THE BEHAVIOR NAME (all caps) or "NONE". No other text."""

        try:
            ctx = SimpleContext()
            model_options = {
                "temperature": 0.3,  # Lower temp for deterministic parsing
                "system_prompt": "You are a helpful assistant. Respond with ONLY a behavior name or NONE.",
            }
            action = CBlock(prompt)

            mot, gen_ctx = await mfuncs.aact(
                action, ctx, self.backend, strategy=None, model_options=model_options
            )

            response_text = await mot.avalue()
            response_text = response_text.strip().upper()

            # Try to match a DogState
            for state in DogState:
                if state.value.upper() in response_text:
                    return state

            return None
        except Exception as e:
            print(f"Command interpretation error: {e}")
            return None

    async def generate_response_to_user(
        self, user_input: str, current_state: DogState, metrics: DogStateMetrics
    ) -> str:
        """
        Generate a text description of how the dog responds to user.

        Args:
            user_input: What the user said
            current_state: Dog's current state
            metrics: Dog's metrics

        Returns:
            Response description (what the dog does)
        """
        energy_level = metrics.energy_level
        state_name = current_state.value

        prompt = f"""You are describing how a toy dog responds to a user.

USER SAID: "{user_input}"
DOG STATE: {state_name}
DOG ENERGY: {energy_level}%

How does the dog react? Write a short 1-2 sentence response describing the dog's reaction behavior.
Be natural and varied based on the dog's state and energy. Keep it under 50 words."""

        try:
            ctx = SimpleContext()
            model_options = {
                "temperature": 0.8,  # Higher temp for varied responses
                "system_prompt": "You are a helpful assistant describing dog behavior.",
            }
            action = CBlock(prompt)

            mot, gen_ctx = await mfuncs.aact(
                action, ctx, self.backend, strategy=None, model_options=model_options
            )

            response = await mot.avalue()
            return response.strip()
        except Exception as e:
            print(f"Response generation error: {e}")
            return f"The dog {state_name}s."

    def _describe_time_of_day(self, hour: int) -> str:
        """Get human-readable time of day."""
        if hour < 6:
            return "night/early morning"
        elif hour < 12:
            return "morning"
        elif hour < 14:
            return "midday"
        elif hour < 18:
            return "afternoon"
        elif hour < 21:
            return "evening"
        else:
            return "late evening"

    def _parse_behavior_response(self, response_text: str) -> Optional[Tuple[DogState, int]]:
        """
        Parse LLM JSON response to extract behavior and duration.

        Args:
            response_text: LLM response (should be JSON)

        Returns:
            (DogState, duration_ms) or None
        """
        try:
            # Try to extract JSON
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start < 0 or json_end <= json_start:
                return None

            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)

            # Extract behavior
            behavior_str = data.get("behavior", "").upper()
            duration = data.get("duration_ms", 10000)

            # Match to DogState
            for state in DogState:
                if state.value.upper() == behavior_str:
                    # Validate duration is within reasonable range
                    duration = max(1000, min(duration, 3600000))  # 1s to 1h
                    return state, int(duration)

            return None
        except json.JSONDecodeError:
            return None
        except Exception as e:
            print(f"Parse error: {e}")
            return None


class DogBehaviorAnalyzer:
    """Analyze patterns and generate behavior insights."""

    def __init__(self):
        """Initialize analyzer."""
        self.behavior_history = []
        self.max_history = 100

    def record_behavior(self, state: DogState, duration_ms: int, metrics: DogStateMetrics):
        """Record executed behavior for analysis."""
        self.behavior_history.append({
            "state": state.value,
            "duration_ms": duration_ms,
            "energy": metrics.energy_level,
            "hunger": metrics.hunger_level,
            "attention": metrics.attention_level,
        })

        # Keep only recent history
        if len(self.behavior_history) > self.max_history:
            self.behavior_history.pop(0)

    def get_behavior_stats(self) -> dict:
        """Get statistics on behavior patterns."""
        if not self.behavior_history:
            return {}

        stats = {}
        for entry in self.behavior_history:
            state = entry["state"]
            if state not in stats:
                stats[state] = {"count": 0, "total_duration": 0, "avg_energy": 0}

            stats[state]["count"] += 1
            stats[state]["total_duration"] += entry["duration_ms"]
            stats[state]["avg_energy"] += entry["energy"]

        # Calculate averages
        for state in stats:
            count = stats[state]["count"]
            stats[state]["avg_energy"] = stats[state]["avg_energy"] / count
            stats[state]["avg_duration"] = stats[state]["total_duration"] / count

        return stats

    def identify_preferences(self) -> dict:
        """Identify what behaviors the dog prefers."""
        stats = self.get_behavior_stats()
        if not stats:
            return {}

        # Sort by frequency
        preferences = sorted(stats.items(), key=lambda x: x[1]["count"], reverse=True)

        return {
            "most_common": preferences[0][0] if preferences else None,
            "least_common": preferences[-1][0] if preferences else None,
            "stats": stats,
        }
