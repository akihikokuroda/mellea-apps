from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import asyncio
import json
from pathlib import Path
from typing import Optional

from dogmove import MotionFrame


class DogHardwareInterface(ABC):
    """Abstract base class for dog hardware control."""

    @abstractmethod
    async def set_motor_angle(self, motor_id: str, angle_degrees: float, duration_ms: int):
        """Set a single motor to a specific angle."""
        pass

    @abstractmethod
    async def execute_motion_sequence(self, frames: list[MotionFrame]):
        """Execute a sequence of motion frames with precise timing."""
        pass

    @abstractmethod
    async def emergency_stop(self):
        """Emergency stop all motors."""
        pass

    @abstractmethod
    async def get_status(self) -> dict:
        """Get current hardware status."""
        pass


class SimulationHardwareInterface(DogHardwareInterface):
    """Simulates hardware by logging to JSON files."""

    def __init__(self, log_dir: str = "dog_simulation_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.start_time = datetime.now()
        self.motion_log: list[dict] = []
        self.is_stopped = False

    async def set_motor_angle(self, motor_id: str, angle_degrees: float, duration_ms: int):
        """Log motor command."""
        if self.is_stopped:
            return

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_ms": (datetime.now() - self.start_time).total_seconds() * 1000,
            "motor_id": motor_id,
            "angle_degrees": angle_degrees,
            "duration_ms": duration_ms,
        }
        self.motion_log.append(log_entry)
        await asyncio.sleep(duration_ms / 1000)

    async def execute_motion_sequence(self, frames: list[MotionFrame]):
        """Execute motion frames with simulated timing."""
        if self.is_stopped:
            return

        for i, frame in enumerate(frames):
            if self.is_stopped:
                break

            for motor_id, angle_degrees in frame.motor_angles.items():
                # Calculate time to next frame
                if i + 1 < len(frames):
                    next_timestamp = frames[i + 1].timestamp_ms
                else:
                    next_timestamp = frame.timestamp_ms + 100

                duration = next_timestamp - frame.timestamp_ms

                log_entry = {
                    "frame_index": i,
                    "timestamp": datetime.now().isoformat(),
                    "elapsed_ms": (datetime.now() - self.start_time).total_seconds() * 1000,
                    "motor_id": motor_id,
                    "angle_degrees": angle_degrees,
                    "frame_duration_ms": duration,
                }
                self.motion_log.append(log_entry)

            # Wait for frame timing
            if i + 1 < len(frames):
                wait_time = (frames[i + 1].timestamp_ms - frame.timestamp_ms) / 1000
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

    async def emergency_stop(self):
        """Stop execution."""
        self.is_stopped = True

    async def get_status(self) -> dict:
        """Return simulation status."""
        return {
            "interface_type": "simulation",
            "is_running": not self.is_stopped,
            "log_entries": len(self.motion_log),
            "elapsed_time_ms": (datetime.now() - self.start_time).total_seconds() * 1000,
        }

    def save_log(self, filename: Optional[str] = None):
        """Save motion log to JSON file."""
        if filename is None:
            filename = f"motion_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.log_dir / filename
        with open(filepath, "w") as f:
            json.dump(self.motion_log, f, indent=2)
        print(f"Motion log saved to {filepath}")


class ServoControlInterface(DogHardwareInterface):
    """Real hardware control via servo/motor driver.

    Requires:
    - Raspberry Pi or similar with GPIO
    - Motor driver (e.g., PCA9685 PWM servo controller)
    - Connected servos to each motor channel
    """

    def __init__(self, i2c_address: int = 0x40, frequency: int = 50):
        """
        Initialize servo controller.

        Args:
            i2c_address: I2C address of PWM controller
            frequency: PWM frequency in Hz (typically 50Hz for servos)
        """
        self.i2c_address = i2c_address
        self.frequency = frequency
        self.motor_map = {}
        self.is_stopped = False

        # Motor channel mapping (customize for your hardware)
        self.motor_channels = {
            "head": 0,
            "tail": 1,
            "left_ear": 2,
            "right_ear": 3,
            "jaw": 4,
            "hip": 5,
            "front_paw": 6,
            "rear_legs": 7,
            "front_left": 8,
            "front_right": 9,
            "back_left": 10,
            "back_right": 11,
        }

    async def _initialize_pwm_driver(self):
        """Initialize PWM driver (PCA9685 or similar)."""
        try:
            from adafruit_pca9685 import PCA9685
            import board
            import busio

            i2c = busio.I2C(board.SCL, board.SDA)
            self.pwm = PCA9685(i2c, address=self.i2c_address)
            self.pwm.frequency = self.frequency
            return True
        except ImportError:
            print("Warning: Adafruit PCA9685 library not installed")
            return False
        except Exception as e:
            print(f"Error initializing PWM driver: {e}")
            return False

    async def set_motor_angle(self, motor_id: str, angle_degrees: float, duration_ms: int):
        """
        Set a single motor to a specific angle.

        Args:
            motor_id: Name of motor (e.g., "tail", "head")
            angle_degrees: Target angle in degrees (0-180 typical)
            duration_ms: Time to reach target in milliseconds
        """
        if self.is_stopped or not hasattr(self, "pwm"):
            return

        if motor_id not in self.motor_channels:
            print(f"Warning: Unknown motor {motor_id}")
            return

        channel = self.motor_channels[motor_id]

        # Convert angle to PWM value (typically 1000-2000 microseconds for 0-180 degrees)
        # Standard servo: 1000us = 0°, 2000us = 180°
        pulse_width = int(1000 + (angle_degrees / 180) * 1000)

        # Set PWM pulse
        self.pwm.channels[channel].duty_cycle = pulse_width

        # Wait for duration
        await asyncio.sleep(duration_ms / 1000)

    async def execute_motion_sequence(self, frames: list[MotionFrame]):
        """Execute motion frames with precise timing on real hardware."""
        if self.is_stopped or not hasattr(self, "pwm"):
            return

        start_time = datetime.now()

        for i, frame in enumerate(frames):
            if self.is_stopped:
                break

            for motor_id, angle_degrees in frame.motor_angles.items():
                await self.set_motor_angle(motor_id, angle_degrees, 0)

            # Wait for frame timing
            if i + 1 < len(frames):
                next_frame_time = frames[i + 1].timestamp_ms / 1000
                current_time = (datetime.now() - start_time).total_seconds()
                wait_time = next_frame_time - current_time

                if wait_time > 0:
                    await asyncio.sleep(wait_time)

    async def emergency_stop(self):
        """Stop all motors."""
        self.is_stopped = True
        if hasattr(self, "pwm"):
            for channel in self.pwm.channels:
                channel.duty_cycle = 0

    async def get_status(self) -> dict:
        """Return hardware status."""
        return {
            "interface_type": "servo_control",
            "is_running": not self.is_stopped,
            "i2c_address": hex(self.i2c_address),
            "frequency_hz": self.frequency,
            "motor_count": len(self.motor_channels),
        }


def create_hardware_interface(interface_type: str = "simulation", **kwargs) -> DogHardwareInterface:
    """Factory function to create appropriate hardware interface."""
    if interface_type == "simulation":
        return SimulationHardwareInterface(**kwargs)
    elif interface_type == "servo":
        return ServoControlInterface(**kwargs)
    else:
        raise ValueError(f"Unknown interface type: {interface_type}")
