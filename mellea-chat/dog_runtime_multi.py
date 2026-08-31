"""Multi-dog 24/7 runtime coordinator for concurrent dog behavior execution.

Manages multiple dogs running simultaneously with independent behaviors,
shared hardware interface, and unified logging with dog_id tags.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any
from collections import deque

from dog_scheduler_multi import MultiDogScheduler, DogInstance
from dog_orchestrator_multi import MultiDogOrchestrator
from dog_interaction_detector import InteractionDetector
from hardware_interface import DogHardwareInterface, create_hardware_interface
from dogmove import MotionFrame


class MultiDogCommandQueue:
    """Command queue supporting per-dog and broadcast commands."""

    def __init__(self, maxsize: int = 50):
        self.queue = deque(maxlen=maxsize)

    def put(self, command: str, dog_id: Optional[str] = None):
        """
        Add command to queue.

        Args:
            command: Command string (e.g., "play", "stop", "status")
            dog_id: Target dog ID, or None for broadcast
        """
        if dog_id:
            self.queue.append((dog_id, command))
        else:
            self.queue.append(("all", command))

    def get(self) -> Optional[tuple[str, str]]:
        """
        Get next command from queue.

        Returns:
            Tuple of (dog_id or "all", command), or None if empty
        """
        if self.queue:
            return self.queue.popleft()
        return None

    def has_data(self) -> bool:
        """Check if queue has commands."""
        return len(self.queue) > 0

    def clear(self):
        """Clear all queued commands."""
        self.queue.clear()


class MultiDogRuntime:
    """Coordinates 24/7 operation for multiple dogs running concurrently."""

    def __init__(
        self,
        dog_count: int = 2,
        hardware_interface: Optional[DogHardwareInterface] = None,
        log_dir: str = "dog_runtime_logs_multi",
        enable_logging: bool = True,
    ):
        """
        Initialize multi-dog runtime.

        Args:
            dog_count: Number of dogs to manage
            hardware_interface: Shared hardware interface (simulation or real)
            log_dir: Directory for runtime logs
            enable_logging: Whether to log state changes
        """
        self.scheduler = MultiDogScheduler(dog_count=dog_count)
        self.orchestrator = MultiDogOrchestrator()
        self.interaction_detector = InteractionDetector()
        self.hardware = hardware_interface or create_hardware_interface("simulation")
        self.command_queue = MultiDogCommandQueue()

        self.log_dir = Path(log_dir)
        self.enable_logging = enable_logging
        if self.enable_logging:
            self.log_dir.mkdir(exist_ok=True)

        self.is_running = False
        self.state_log: list[dict] = []
        self.start_time = datetime.now()
        self.cycle_count = 0

    async def _log_state(self, event: str, dog_id: str, details: dict = None):
        """Log state change with dog_id tag."""
        if not self.enable_logging:
            return

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_ms": (datetime.now() - self.start_time).total_seconds() * 1000,
            "event": event,
            "dog_id": dog_id,
            "cycle": self.cycle_count,
        }

        dog = self.scheduler.get_dog(dog_id)
        if dog:
            log_entry["dog_state"] = dog.current_state.value
            log_entry["metrics"] = {
                "energy": dog.get_metrics().energy_level,
                "hunger": dog.get_metrics().hunger_level,
                "attention": dog.get_metrics().attention_level,
            }

        if details:
            log_entry.update(details)

        self.state_log.append(log_entry)
        print(f"[{event}] dog_id={dog_id} - {dog.describe_state() if dog else 'N/A'}")

    async def _process_commands(self):
        """Process pending commands from queue."""
        while self.command_queue.has_data():
            target, command = self.command_queue.get()

            if command == "stop" or command == "quit":
                self.is_running = False
                await self.hardware.emergency_stop()
                print("[COMMAND] Stop requested")
                break

            elif command == "status":
                all_states = self.scheduler.get_all_states()
                all_metrics = self.scheduler.get_all_metrics()
                for dog_id in self.scheduler.get_all_dogs().keys():
                    state_str = all_states.get(dog_id, "unknown")
                    metrics = all_metrics.get(dog_id)
                    print(f"[STATUS] {state_str}")
                    if metrics:
                        print(f"         Energy: {metrics.energy_level}%, "
                              f"Hunger: {metrics.hunger_level}%, "
                              f"Attention: {metrics.attention_level}%")

    async def _run_dog_cycle(self, dog_id: str, cycle_num: int):
        """
        Execute one behavior cycle for a single dog.

        Runs independently in its own task, sleeps for behavior duration.
        Handles both solo and interactive behaviors.

        Args:
            dog_id: Dog identifier
            cycle_num: Cycle number for logging
        """
        try:
            dog = self.scheduler.get_dog(dog_id)
            if not dog:
                return

            self.cycle_count = cycle_num

            # Get next behavior
            next_state, duration_ms = dog.get_next_action()
            dog.current_state = next_state

            await self._log_state(
                f"cycle_{next_state.value}",
                dog_id,
                {
                    "duration_ms": duration_ms,
                    "interaction_target": dog.scheduler.metrics.interaction_target,
                    "interaction_type": dog.scheduler.metrics.interaction_type,
                },
            )

            # Get movement frames for this dog
            dog_behaviors = {dog_id: next_state}
            intensity_map = {dog_id: 0.7}
            duration_map = {dog_id: duration_ms}

            # Include interaction pair info if dog is in an interaction
            interaction_pairs = None
            if (
                dog.scheduler.metrics.interaction_target
                and dog.scheduler.metrics.interaction_type
            ):
                interaction_target = dog.scheduler.metrics.interaction_target
                interaction_type = dog.scheduler.metrics.interaction_type
                # Only include pair once (avoid duplicates)
                if dog_id < interaction_target:
                    interaction_pairs = [(dog_id, interaction_target, interaction_type)]

            movements = self.orchestrator.get_movements_for_all_dogs(
                dog_behaviors=dog_behaviors,
                intensity_map=intensity_map,
                duration_map=duration_map,
                interaction_pairs=interaction_pairs,
            )

            if movements:
                await self.hardware.execute_motion_sequence(movements)

            # Update state metrics
            elapsed_map = {dog_id: duration_ms}
            self.scheduler.update_all_states(elapsed_map)

        except Exception as e:
            await self._log_state("error", dog_id, {"error": str(e)})
            print(f"Error in dog {dog_id} cycle: {e}")

    async def run_cycle(self, cycle_num: int):
        """
        Execute one cycle for all dogs concurrently.

        Each dog runs its own behavior independently in parallel.
        Detects and initiates dog-to-dog interactions.

        Args:
            cycle_num: Cycle number for logging
        """
        # Process any pending commands
        await self._process_commands()

        if not self.is_running:
            return False

        # Detect nearby dogs and suggest interactions
        nearby_dogs = self.interaction_detector.detect_nearby_dogs(self.scheduler)
        interaction_suggestions = self.interaction_detector.suggest_interactions(
            self.scheduler, nearby_dogs, interaction_probability=0.25
        )

        # Initiate suggested interactions
        for dog_1_id, dog_2_id, interaction_type in interaction_suggestions:
            if self.scheduler.initiate_interaction(dog_1_id, dog_2_id, interaction_type):
                await self._log_state(
                    "interaction_start",
                    dog_1_id,
                    {
                        "partner": dog_2_id,
                        "interaction_type": interaction_type,
                    },
                )
                await self._log_state(
                    "interaction_start",
                    dog_2_id,
                    {
                        "partner": dog_1_id,
                        "interaction_type": interaction_type,
                    },
                )

        # Run all dogs' cycles concurrently
        dog_ids = list(self.scheduler.get_all_dogs().keys())
        tasks = [self._run_dog_cycle(dog_id, cycle_num) for dog_id in dog_ids]
        await asyncio.gather(*tasks)

        # Update interaction states
        self.scheduler.update_interactions(elapsed_map={dog_id: 3000 for dog_id in dog_ids})

        return True

    async def run(self, duration_seconds: int = 60, cycle_limit: Optional[int] = None):
        """
        Run multi-dog 24/7 loop for specified duration.

        Args:
            duration_seconds: Total duration to run
            cycle_limit: Max cycles to run (None for no limit)
        """
        self.is_running = True
        start_time = datetime.now()

        print(f"\n=== Starting Multi-Dog Runtime ===")
        print(f"Dogs: {self.scheduler.get_dog_count()}")
        print(f"Duration: {duration_seconds}s")
        if cycle_limit:
            print(f"Cycle limit: {cycle_limit}")
        print("=" * 35 + "\n")

        cycle_num = 0
        try:
            while self.is_running:
                if cycle_limit and cycle_num >= cycle_limit:
                    break

                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > duration_seconds:
                    break

                await self.run_cycle(cycle_num)
                cycle_num += 1

        except KeyboardInterrupt:
            print("\n[INTERRUPT] Stopping...")
        finally:
            await self.hardware.emergency_stop()
            self.is_running = False
            await self._save_logs()

    def queue_command(self, command: str, dog_id: Optional[str] = None):
        """
        Queue a command for a dog or all dogs.

        Args:
            command: Command string
            dog_id: Target dog, or None for all dogs
        """
        self.command_queue.put(command, dog_id)

    async def _save_logs(self):
        """Save runtime logs to JSON file."""
        if not self.enable_logging or not self.state_log:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"multi_dog_runtime_{timestamp}.json"

        with open(log_file, "w") as f:
            json.dump(self.state_log, f, indent=2)

        print(f"\nLogs saved to {log_file}")

    def get_state_summary(self) -> dict[str, Any]:
        """Get summary of current state for all dogs."""
        return {
            "cycle_count": self.cycle_count,
            "is_running": self.is_running,
            "dogs_count": self.scheduler.get_dog_count(),
            "dogs_states": self.scheduler.get_all_states(),
            "dogs_metrics": {
                dog_id: {
                    "energy": metrics.energy_level,
                    "hunger": metrics.hunger_level,
                    "attention": metrics.attention_level,
                }
                for dog_id, metrics in self.scheduler.get_all_metrics().items()
            },
        }


async def run_multi_dog_24_7(
    interface_type: str = "simulation",
    dog_count: int = 2,
    duration_seconds: int = 60,
    cycle_limit: Optional[int] = None,
) -> MultiDogRuntime:
    """
    Convenience function to run multi-dog system.

    Args:
        interface_type: "simulation" or "servo"
        dog_count: Number of dogs
        duration_seconds: Duration to run
        cycle_limit: Max cycles

    Returns:
        MultiDogRuntime instance
    """
    hardware = create_hardware_interface(interface_type)
    runtime = MultiDogRuntime(
        dog_count=dog_count,
        hardware_interface=hardware,
    )
    await runtime.run(duration_seconds=duration_seconds, cycle_limit=cycle_limit)
    return runtime


if __name__ == "__main__":
    import sys

    # Example: python dog_runtime_multi.py [simulation|servo] [dog_count] [duration_sec] [cycle_limit]
    interface = sys.argv[1] if len(sys.argv) > 1 else "simulation"
    dogs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    duration = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    cycles = int(sys.argv[4]) if len(sys.argv) > 4 else None

    runtime = asyncio.run(run_multi_dog_24_7(
        interface_type=interface,
        dog_count=dogs,
        duration_seconds=duration,
        cycle_limit=cycles,
    ))
