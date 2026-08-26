"""Natural language command parser for user interaction with the dog."""

from dataclasses import dataclass
from typing import Optional, List
from dog_scheduler import DogState


@dataclass
class ParsedCommand:
    """Represents a parsed user command."""
    command_type: str  # "action", "query", "status", "help", "invalid"
    target_state: Optional[DogState] = None
    query_type: Optional[str] = None  # "state", "energy", "hunger", "attention"
    raw_input: str = ""
    confidence: float = 0.8


class CommandParser:
    """Parses natural language commands into structured actions."""

    # Command mappings
    ACTION_KEYWORDS = {
        "play": [DogState.PLAYING, "make the dog play"],
        "sleep": [DogState.SLEEPING, "make the dog sleep"],
        "rest": [DogState.RESTING, "make the dog rest"],
        "eat": [DogState.EATING, "make the dog eat"],
        "feed": [DogState.EATING, "feed the dog"],
        "explore": [DogState.EXPLORING, "make the dog explore"],
        "scratch": [DogState.SCRATCHING, "make the dog scratch"],
        "groom": [DogState.GROOMING, "make the dog groom"],
        "alert": [DogState.ALERT, "make the dog alert"],
        "run": [DogState.PLAYING, "make the dog run"],
        "nap": [DogState.SLEEPING, "make the dog nap"],
        "sniff": [DogState.EXPLORING, "make the dog sniff"],
        "walk": [DogState.EXPLORING, "make the dog walk"],
    }

    GREETING_KEYWORDS = ["hello", "hi", "hey", "greetings", "howdy"]
    QUERY_KEYWORDS = {
        "energy": "energy",
        "tired": "energy",
        "hungry": "hunger",
        "hunger": "hunger",
        "attention": "attention",
        "happy": "attention",
        "state": "state",
        "doing": "state",
        "status": "state",
    }

    STATUS_KEYWORDS = ["status", "what is", "how is", "tell me about"]
    HELP_KEYWORDS = ["help", "commands", "what can", "instructions"]
    STOP_KEYWORDS = ["stop", "quit", "exit", "bye"]

    def parse(self, user_input: str) -> ParsedCommand:
        """
        Parse user input into a command.

        Args:
            user_input: Natural language input from user

        Returns:
            ParsedCommand with command details
        """
        user_input_lower = user_input.lower().strip()

        # Check for stop/exit
        if any(kw in user_input_lower for kw in self.STOP_KEYWORDS):
            return ParsedCommand(
                command_type="stop",
                raw_input=user_input,
                confidence=0.95
            )

        # Check for help
        if any(kw in user_input_lower for kw in self.HELP_KEYWORDS):
            return ParsedCommand(
                command_type="help",
                raw_input=user_input,
                confidence=0.95
            )

        # Check for status queries (before greetings since "what is" contains important context)
        if any(kw in user_input_lower for kw in self.STATUS_KEYWORDS):
            return ParsedCommand(
                command_type="status",
                raw_input=user_input,
                confidence=0.85
            )

        # Check for queries (before greetings since query keywords are more specific)
        for keyword, query_type in self.QUERY_KEYWORDS.items():
            if keyword in user_input_lower:
                return ParsedCommand(
                    command_type="query",
                    query_type=query_type,
                    raw_input=user_input,
                    confidence=0.8
                )

        # Check for greetings (after more specific patterns)
        if any(kw in user_input_lower for kw in self.GREETING_KEYWORDS):
            return ParsedCommand(
                command_type="greeting",
                raw_input=user_input,
                confidence=0.9
            )

        # Check for actions
        for keyword, (state, _) in self.ACTION_KEYWORDS.items():
            if keyword in user_input_lower:
                return ParsedCommand(
                    command_type="action",
                    target_state=state,
                    raw_input=user_input,
                    confidence=0.85
                )

        # Default to invalid
        return ParsedCommand(
            command_type="invalid",
            raw_input=user_input,
            confidence=0.0
        )

    def get_help_text(self) -> str:
        """Get help text for available commands."""
        help_text = """
🐕 DOG COMMANDS - Available Actions:

BEHAVIOR COMMANDS:
  - "Make the dog play" - Play active game
  - "Let the dog sleep" - Take a nap
  - "Dog should rest" - Relax and recover
  - "Feed the dog" - Eating time
  - "Dog explore" - Walk around exploring
  - "Scratch" / "Groom" - Self-care

STATUS QUERIES:
  - "Is the dog tired?" - Check energy level
  - "Is the dog hungry?" - Check hunger
  - "Dog status" - Full state description
  - "What is the dog doing?" - Current behavior
  - "How's the dog feeling?" - Overall status

INFO:
  - "Help" - Show this message
  - "Stop" / "Quit" - Exit program
"""
        return help_text


class ResponseGenerator:
    """Generates natural responses to user commands."""

    @staticmethod
    def get_action_confirmation(state: DogState) -> str:
        """Generate confirmation for an action command."""
        confirmations = {
            DogState.PLAYING: "🎾 The dog gets excited and is ready to play!",
            DogState.SLEEPING: "😴 The dog yawns and settles down for a nap...",
            DogState.RESTING: "😌 The dog relaxes and takes a break.",
            DogState.EXPLORING: "🔍 The dog perks up and starts exploring!",
            DogState.EATING: "🍖 The dog heads over to eat.",
            DogState.SCRATCHING: "🤔 The dog scratches and stretches.",
            DogState.GROOMING: "🧼 The dog starts grooming itself.",
            DogState.ALERT: "👀 The dog becomes alert and watchful.",
        }
        return confirmations.get(state, f"The dog starts {state.value}ing.")

    @staticmethod
    def get_query_response(query_type: str, metrics) -> str:
        """Generate response to a query."""
        # Handle both dict and object metrics
        energy = metrics.get("energy_level") if isinstance(metrics, dict) else metrics.energy_level
        hunger = metrics.get("hunger_level") if isinstance(metrics, dict) else metrics.hunger_level
        attention = metrics.get("attention_level") if isinstance(metrics, dict) else metrics.attention_level

        if query_type == "energy":
            if energy > 70:
                return f"⚡ The dog is very energetic! ({energy}% energy)"
            elif energy > 40:
                return f"😊 The dog feels good. ({energy}% energy)"
            else:
                return f"😴 The dog is getting tired. ({energy}% energy)"

        elif query_type == "hunger":
            if hunger > 70:
                return f"🍗 The dog is hungry! ({hunger}% hunger)"
            elif hunger > 40:
                return f"😋 The dog could eat something. ({hunger}% hunger)"
            else:
                return f"😊 The dog is satisfied. ({hunger}% hunger)"

        elif query_type == "attention":
            if attention > 60:
                return f"👋 The dog wants your attention! ({attention}% focus)"
            elif attention > 30:
                return f"🐕 The dog notices you. ({attention}% focus)"
            else:
                return f"😌 The dog is focused on own activity. ({attention}% focus)"

        elif query_type == "state":
            return f"The dog is busy with its activities."

        return "The dog looks fine!"

    @staticmethod
    def get_status_response(current_state: DogState, metrics) -> str:
        """Generate full status response."""
        # Handle both dict and object metrics
        energy = metrics.get("energy_level") if isinstance(metrics, dict) else metrics.energy_level
        hunger = metrics.get("hunger_level") if isinstance(metrics, dict) else metrics.hunger_level
        attention = metrics.get("attention_level") if isinstance(metrics, dict) else metrics.attention_level

        energy_bar = ResponseGenerator._make_bar(energy, "⚡")
        hunger_bar = ResponseGenerator._make_bar(hunger, "🍗")
        attention_bar = ResponseGenerator._make_bar(attention, "👀")

        status = f"""
🐕 DOG STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current Behavior: {current_state.value.upper()}

Energy:     {energy_bar} {energy}%
Hunger:     {hunger_bar} {hunger}%
Attention:  {attention_bar} {attention}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return status

    @staticmethod
    def _make_bar(percentage: int, emoji: str, length: int = 10) -> str:
        """Create a simple bar chart."""
        filled = int((percentage / 100) * length)
        bar = emoji * filled + "░" * (length - filled)
        return f"[{bar}]"

    @staticmethod
    def get_greeting_response() -> str:
        """Response to greeting."""
        responses = [
            "🐕 The dog perks up excitedly and tail wags!",
            "🐕 Woof! The dog bounds over to greet you.",
            "🐕 The dog's eyes light up as it sees you!",
            "🐕 Tail wagging enthusiastically! The dog is happy to see you.",
            "🐕 The dog does a little spin and barks happily!",
        ]
        import random
        return random.choice(responses)

    @staticmethod
    def get_invalid_response() -> str:
        """Response to invalid command."""
        responses = [
            "I'm not sure what you mean. Try 'help' for commands.",
            "The dog tilts its head, confused.",
            "🤔 I didn't understand that. Type 'help' for options.",
            "The dog gives you a puzzled look.",
        ]
        import random
        return random.choice(responses)

    @staticmethod
    def get_help_response() -> str:
        """Response when user asks for help."""
        help_text = CommandParser().get_help_text()
        return help_text

    @staticmethod
    def get_stop_response() -> str:
        """Response when user exits."""
        return """
👋 Goodbye! The dog waves its tail as you leave...
See you next time! 🐕
"""


class ConversationManager:
    """Manages conversation state and history."""

    def __init__(self, max_history: int = 50):
        """Initialize conversation manager."""
        self.history: List[dict] = []
        self.max_history = max_history
        self.parser = CommandParser()
        self.responder = ResponseGenerator()

    def add_exchange(self, user_input: str, dog_response: str, command_type: str = ""):
        """Add a user-dog exchange to history."""
        self.history.append({
            "timestamp": self._get_timestamp(),
            "user": user_input,
            "dog": dog_response,
            "command_type": command_type,
        })

        # Keep history trimmed
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_history(self) -> List[dict]:
        """Get conversation history."""
        return self.history

    def get_summary(self) -> dict:
        """Get summary of conversation."""
        if not self.history:
            return {"total_exchanges": 0, "command_types": {}}

        command_counts = {}
        for exchange in self.history:
            cmd_type = exchange.get("command_type", "unknown")
            command_counts[cmd_type] = command_counts.get(cmd_type, 0) + 1

        return {
            "total_exchanges": len(self.history),
            "command_types": command_counts,
            "last_user_message": self.history[-1]["user"] if self.history else None,
        }

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


class InteractiveSession:
    """Manages an interactive session with the user and dog."""

    def __init__(self):
        """Initialize session."""
        self.manager = ConversationManager()
        self.parser = CommandParser()
        self.responder = ResponseGenerator()
        self.is_running = False

    def process_command(self, user_input: str, dog_state, dog_metrics):
        """
        Process a user command and generate response.

        Args:
            user_input: User input text
            dog_state: Current DogState
            dog_metrics: Current DogStateMetrics

        Returns:
            (command_type, target_state_or_query, response_text)
        """
        # Parse command
        parsed = self.parser.parse(user_input)

        # Generate response
        if parsed.command_type == "action":
            response = self.responder.get_action_confirmation(parsed.target_state)
            self.manager.add_exchange(user_input, response, "action")
            return "action", parsed.target_state, response

        elif parsed.command_type == "greeting":
            response = self.responder.get_greeting_response()
            self.manager.add_exchange(user_input, response, "greeting")
            return "greeting", None, response

        elif parsed.command_type == "query":
            response = self.responder.get_query_response(parsed.query_type, dog_metrics)
            self.manager.add_exchange(user_input, response, "query")
            return "query", parsed.query_type, response

        elif parsed.command_type == "status":
            response = self.responder.get_status_response(dog_state, dog_metrics)
            self.manager.add_exchange(user_input, response, "status")
            return "status", None, response

        elif parsed.command_type == "help":
            response = self.responder.get_help_response()
            self.manager.add_exchange(user_input, response, "help")
            return "help", None, response

        elif parsed.command_type == "stop":
            response = self.responder.get_stop_response()
            self.manager.add_exchange(user_input, response, "stop")
            return "stop", None, response

        else:  # invalid
            response = self.responder.get_invalid_response()
            self.manager.add_exchange(user_input, response, "invalid")
            return "invalid", None, response

    def get_conversation_summary(self) -> dict:
        """Get summary of conversation."""
        return self.manager.get_summary()

    def get_history(self) -> List[dict]:
        """Get conversation history."""
        return self.manager.get_history()
