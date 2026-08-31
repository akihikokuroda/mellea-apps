import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any
from collections import deque

from dog_scheduler import DailySchedule, DogState
from hardware_interface import DogHardwareInterface, create_hardware_interface
from dog_behavior_orchestrator import BehaviorOrchestrator
from mellea_dog_behavior import DogBehaviorLLM, DogBehaviorAnalyzer
from dogmove import MotionFrame


class CommandQueue:
    """Thread-safe queue for user commands."""

    def __init__(self, maxsize: int = 10):
        self.queue = deque(maxlen=maxsize)

    def put(self, command: str):
        """Add command to queue."""
        self.queue.append(command)

    def get(self) -> Optional[str]:
        """Get next command from queue."""
        if self.queue:
            return self.queue.popleft()
        return None

    def has_data(self) -> bool:
        """Check if queue has commands."""
        return len(self.queue) > 0

    def clear(self):
        """Clear all queued commands."""
        self.queue.clear()


class DogRuntime:
    """Main 24/7 runtime coordinator for the AI dog."""

    def __init__(
        self,
        hardware_interface: Optional[DogHardwareInterface] = None,
        behavior_orchestrator: Optional[BehaviorOrchestrator] = None,
        llm_advisor: Optional[DogBehaviorLLM] = None,
        log_dir: str = "dog_runtime_logs",
        enable_logging: bool = True,
        use_llm: bool = True,
    ):
        """
        Initialize the dog runtime.

        Args:
            hardware_interface: Hardware interface (simulation or real)
            behavior_orchestrator: Behavior-to-movement mapper
            llm_advisor: LLM-based behavior advisor
            log_dir: Directory for runtime logs
            enable_logging: Whether to log state changes
            use_llm: Whether to use LLM for behavior decisions
        """
        self.scheduler = DailySchedule()
        self.hardware = hardware_interface or create_hardware_interface("simulation")
        self.behavior_orchestrator = behavior_orchestrator or BehaviorOrchestrator()
        self.llm_advisor = llm_advisor if use_llm else None
        self.behavior_analyzer = DogBehaviorAnalyzer()
        self.use_llm = use_llm
        self.command_queue = CommandQueue()

        self.log_dir = Path(log_dir)
        self.enable_logging = enable_logging
        if self.enable_logging:
            self.log_dir.mkdir(exist_ok=True)

        self.is_running = False
        self.state_log: list[dict] = []
        self.start_time = datetime.now()
        self.current_behavior = None

    async def _log_state(self, event: str, details: dict = None):
        """Log state change."""
        if not self.enable_logging:
            return

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_ms": (datetime.now() - self.start_time).total_seconds() * 1000,
            "event": event,
            "dog_state": self.scheduler.current_state.value,
            "metrics": self.scheduler.get_metrics_dict(),
        }

        if details:
            log_entry.update(details)

        self.state_log.append(log_entry)
        print(f"[{event}] {self.scheduler.describe_current_state()}")

    async def _get_next_behavior(self) -> tuple[DogState, int]:
        """Determine next behavior (with optional LLM advisory)."""
        if self.use_llm and self.llm_advisor is not None:
            # Use LLM for intelligent behavior selection
            try:
                from datetime import datetime
                current_hour = datetime.now().hour

                next_state, duration = await self.llm_advisor.decide_behavior(
                    self.scheduler.current_state, self.scheduler.metrics, current_hour
                )
                if next_state:
                    await self._log_state("llm_decision", {
                        "recommended_behavior": next_state.value,
                        "duration_ms": duration,
                    })
                    return next_state, duration
            except Exception as e:
                await self._log_state("llm_error", {"error": str(e)})

        # Fallback to scheduler
        return self.scheduler.get_next_action()

    async def _execute_behavior(self, dog_state: DogState, duration_ms: int):
        """Execute a behavior using the orchestrator."""
        try:
            # Calculate intensity based on metrics
            intensity = (self.scheduler.metrics.energy_level / 100.0) * 0.8 + 0.2

            # Get movements from orchestrator
            movements = self.behavior_orchestrator.get_movements_for_state(
                dog_state, intensity=intensity, duration=duration_ms
            )

            if movements:
                await self._log_state("execute_behavior", {
                    "behavior": dog_state.value,
                    "duration_ms": duration_ms,
                    "frame_count": len(movements),
                    "intensity": intensity,
                })
                await self.hardware.execute_motion_sequence(movements)
            else:
                await self._log_state("behavior_no_movements", {
                    "behavior": dog_state.value,
                    "duration_ms": duration_ms,
                })
        except Exception as e:
            await self._log_state("behavior_error", {
                "behavior": dog_state.value,
                "error": str(e),
            })
            print(f"Error executing behavior: {e}")

    async def _handle_user_command(self, command_text: str):
        """Handle user command (placeholder for Phase 4)."""
        await self._log_state("user_command", {"command": command_text})

        if command_text.lower() in ["status", "state"]:
            status = self.scheduler.describe_current_state()
            await self._log_state("query_response", {"response": status})
        elif command_text.lower() in ["stop", "quit"]:
            self.is_running = False
        else:
            await self._log_state("command_unknown", {"command": command_text})

    async def run_cycle(self) -> bool:
        """
        Run one cycle of the dog's behavior.

        Returns: True if should continue, False to stop.
        """
        # Check for user commands
        while self.command_queue.has_data():
            cmd = self.command_queue.get()
            await self._handle_user_command(cmd)

        if not self.is_running:
            return False

        # Get next behavior
        next_state, duration_ms = await self._get_next_behavior()

        # Execute behavior
        await self._execute_behavior(next_state, duration_ms)

        # Update scheduler state
        self.scheduler.update_state(duration_ms, user_interacted=False, new_state=next_state)

        # Record behavior for analysis
        self.behavior_analyzer.record_behavior(next_state, duration_ms, self.scheduler.metrics)

        return True

    async def run(self, duration_seconds: Optional[float] = None, cycle_limit: Optional[int] = None):
        """
        Run the 24/7 dog runtime.

        Args:
            duration_seconds: Run for specified duration, then stop
            cycle_limit: Stop after specified number of cycles
        """
        self.is_running = True
        await self._log_state("runtime_start", {
            "hardware_type": await self.hardware.get_status(),
        })

        cycle_count = 0

        try:
            while self.is_running:
                # Check stop conditions
                if cycle_limit and cycle_count >= cycle_limit:
                    await self._log_state("cycle_limit_reached", {"cycles": cycle_count})
                    break

                if duration_seconds:
                    elapsed = (datetime.now() - self.start_time).total_seconds()
                    if elapsed >= duration_seconds:
                        await self._log_state("duration_limit_reached", {"seconds": elapsed})
                        break

                # Run one cycle
                should_continue = await self.run_cycle()
                if not should_continue:
                    break

                cycle_count += 1

        except KeyboardInterrupt:
            await self._log_state("interrupted", {"cycles": cycle_count})
        except Exception as e:
            await self._log_state("error", {"error": str(e)})
            raise
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Gracefully shutdown the runtime."""
        self.is_running = False
        await self.hardware.emergency_stop()
        await self._log_state("runtime_shutdown", {
            "total_cycles": len(self.state_log),
        })
        self.save_logs()

    def save_logs(self):
        """Save all logs to JSON files."""
        if not self.enable_logging:
            return

        # Save state log
        state_log_file = self.log_dir / f"state_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(state_log_file, "w") as f:
            json.dump(self.state_log, f, indent=2)
        print(f"State log saved to {state_log_file}")

        # Save hardware log if it's a simulation
        if hasattr(self.hardware, "save_log"):
            self.hardware.save_log()

    def queue_command(self, command: str):
        """Queue a user command for processing."""
        self.command_queue.put(command)

    def get_current_status(self) -> dict:
        """Get current runtime status."""
        return {
            "is_running": self.is_running,
            "elapsed_ms": (datetime.now() - self.start_time).total_seconds() * 1000,
            "dog_state": self.scheduler.current_state.value,
            "metrics": self.scheduler.get_metrics_dict(),
            "queued_commands": self.command_queue.queue,
        }


async def run_dog_24_7(
    interface_type: str = "simulation",
    duration_seconds: Optional[float] = None,
    cycle_limit: Optional[int] = None,
):
    """
    Convenience function to start the dog runtime.

    Args:
        interface_type: "simulation" or "servo"
        duration_seconds: Optional duration limit
        cycle_limit: Optional cycle limit
    """
    hardware = create_hardware_interface(interface_type)
    runtime = DogRuntime(hardware_interface=hardware)

    await runtime.run(duration_seconds=duration_seconds, cycle_limit=cycle_limit)

    return runtime


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    interface_type = sys.argv[1] if len(sys.argv) > 1 else "simulation"
    cycle_limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"Starting dog runtime (interface: {interface_type}, cycles: {cycle_limit or 'unlimited'})")

    # Run
    runtime = asyncio.run(run_dog_24_7(interface_type=interface_type, cycle_limit=cycle_limit))
