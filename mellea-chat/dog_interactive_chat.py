"""Interactive chat interface for the 24/7 AI dog system."""

import asyncio
from typing import Optional
from datetime import datetime

from dog_scheduler import DailySchedule, DogState
from dog_behavior_orchestrator import BehaviorOrchestrator
from mellea_dog_behavior import DogBehaviorLLM
from hardware_interface import create_hardware_interface
from dog_runtime import DogRuntime
from dog_command_parser import InteractiveSession


class DogInteractiveChatUI:
    """Interactive chat UI for the dog system."""

    def __init__(
        self,
        hardware_type: str = "simulation",
        use_llm: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize interactive UI.

        Args:
            hardware_type: "simulation" or "servo"
            use_llm: Whether to use LLM for behavior decisions
            verbose: Print debug information
        """
        self.hardware_type = hardware_type
        self.use_llm = use_llm
        self.verbose = verbose

        # Initialize components
        self.hardware = create_hardware_interface(hardware_type)
        self.orchestrator = BehaviorOrchestrator()
        self.llm_advisor = DogBehaviorLLM() if use_llm else None

        self.runtime = DogRuntime(
            hardware_interface=self.hardware,
            behavior_orchestrator=self.orchestrator,
            llm_advisor=self.llm_advisor,
            enable_logging=True,
            use_llm=use_llm,
        )

        self.session = InteractiveSession()
        self.is_running = False

    async def chat_loop(self):
        """Main interactive chat loop."""
        print("\n" + "=" * 60)
        print("🐕 24/7 AI DOG - INTERACTIVE CHAT")
        print("=" * 60)
        print("\nWelcome! Type 'help' for commands, 'quit' to exit.\n")

        self.is_running = True
        background_task = None

        try:
            # Start background behavior execution
            background_task = asyncio.create_task(self._background_behavior_loop())

            while self.is_running:
                # Get user input
                user_input = await self._get_user_input()

                if not user_input:
                    continue

                # Get current dog state
                status = self.runtime.get_current_status()
                current_state = DogState(status["dog_state"])
                metrics = status["metrics"]

                # Process command
                cmd_type, target, response = self.session.process_command(
                    user_input, current_state, metrics
                )

                # Print response
                print(f"\n🐕 Dog: {response}\n")

                # Handle action commands
                if cmd_type == "action":
                    await self._execute_user_action(target)

                # Handle stop
                if cmd_type == "stop":
                    self.is_running = False

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user.")
        finally:
            if background_task:
                background_task.cancel()
                try:
                    await background_task
                except asyncio.CancelledError:
                    pass
            self.runtime.save_logs()

    async def _background_behavior_loop(self):
        """Run background behavior updates."""
        try:
            while self.is_running:
                # Get next behavior (LLM or scheduler)
                next_state, duration = await self.runtime._get_next_behavior()

                # Log decision
                status = self.runtime.get_current_status()
                current_state = DogState(status["dog_state"])

                if current_state != next_state:
                    print(f"\n[Dog transitions to {next_state.value}]")

                # Execute behavior
                await self.runtime._execute_behavior(next_state, duration)

                # Update scheduler
                self.runtime.scheduler.update_state(
                    duration, user_interacted=False, new_state=next_state
                )

                # Wait for next cycle
                await asyncio.sleep(min(duration / 1000, 3))  # Cap at 3 seconds for responsiveness

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self.verbose:
                print(f"Background error: {e}")

    async def _get_user_input(self) -> str:
        """Get user input asynchronously."""
        # Use run_in_executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        user_input = await loop.run_in_executor(None, input, "You: ")
        return user_input.strip()

    async def _execute_user_action(self, target_state: DogState):
        """Execute a user-requested action."""
        if target_state:
            print(f"\n[Executing {target_state.value}...]")
            # Queue the action
            self.runtime.queue_command(target_state.value)

    def get_session_summary(self) -> dict:
        """Get summary of the chat session."""
        return {
            "conversation": self.session.get_conversation_summary(),
            "runtime": {
                "total_state_events": len(self.runtime.state_log),
                "hardware_type": self.hardware_type,
                "llm_enabled": self.use_llm,
            },
        }


async def run_interactive_chat(
    hardware_type: str = "simulation",
    use_llm: bool = True,
    verbose: bool = False,
):
    """
    Run interactive chat session.

    Args:
        hardware_type: "simulation" or "servo"
        use_llm: Whether to use LLM for decisions
        verbose: Print debug info
    """
    ui = DogInteractiveChatUI(
        hardware_type=hardware_type,
        use_llm=use_llm,
        verbose=verbose,
    )

    await ui.chat_loop()

    return ui


async def run_demo_session():
    """Run a demo session with predefined inputs."""
    print("\n" + "=" * 60)
    print("🐕 24/7 AI DOG - DEMO SESSION")
    print("=" * 60 + "\n")

    ui = DogInteractiveChatUI(use_llm=False)  # Use scheduler for demo

    # Demo commands
    demo_commands = [
        ("Hello dog!", "greeting"),
        ("What is the dog doing?", "status"),
        ("Make the dog play!", "action"),
        ("Is the dog tired?", "query"),
        ("Dog status", "status"),
    ]

    print("Demo Mode: Running predefined commands\n")

    for user_input, cmd_type in demo_commands:
        print(f"\nYou: {user_input}")

        status = ui.runtime.get_current_status()
        current_state = DogState(status["dog_state"])
        metrics = status["metrics"]

        cmd_type_result, target, response = ui.session.process_command(
            user_input, current_state, metrics
        )

        print(f"🐕 Dog: {response}")

        if cmd_type_result == "action":
            print(f"[Executing {target.value}...]")

        await asyncio.sleep(0.5)

    # Print summary
    print("\n" + "=" * 60)
    print("Demo Summary:")
    summary = ui.get_session_summary()
    print(f"  Conversations: {summary['conversation']['total_exchanges']}")
    print(f"  Runtime events: {summary['runtime']['total_state_events']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # Run demo mode
        asyncio.run(run_demo_session())
    else:
        # Run interactive mode
        asyncio.run(run_interactive_chat(use_llm=False))  # Set to True if Ollama is running
