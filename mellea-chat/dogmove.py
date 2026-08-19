"""Toy dog movement functions for animating puppet-like motions.

This module provides typed function signatures for core toy dog movements,
parameterized by intensity, duration, and motion characteristics.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class WagStyle(Enum):
    """Tail wag intensity levels."""

    FAST_LOOSE = "fast_loose"
    SLOW_WIDE = "slow_wide"


class WalkGait(Enum):
    """Locomotion movement styles."""

    STEADY_WALK = "steady_walk"
    FAST_TROT = "fast_trot"


class EarMotion(Enum):
    """Ear movement patterns."""

    VERTICAL = "vertical"
    FLARING = "flaring"


@dataclass
class MotionFrame:
    """Single frame of motion with timing and multi-motor angle data.

    Attributes:
        timestamp_ms: Frame timing in milliseconds.
        motor_angles: Mapping of motor IDs to target angles in degrees.
    """

    timestamp_ms: float
    motor_angles: dict[str, float]


def tail_wag(
    style: WagStyle,
    duration_ms: float,
    swing_arc_degrees: float = 45.0,
    motor_id: str = "tail",
) -> list[MotionFrame]:
    """Generate a side-to-side sweeping tail motion.

    Args:
        style: Wag style (fast_loose for excitement, slow_wide for relaxation).
        duration_ms: Total motion duration in milliseconds.
        swing_arc_degrees: Maximum angle from center in degrees.
        motor_id: Identifier for the tail servo motor.

    Returns:
        List of MotionFrame objects defining the tail path.
    """
    frames = []
    center_angle = 0.0

    if style == WagStyle.FAST_LOOSE:
        # Fast excited wag: quick oscillations, loose movement
        num_cycles = 4
        cycle_duration = duration_ms / num_cycles
        steps_per_cycle = 8

        for cycle in range(num_cycles):
            for step in range(steps_per_cycle):
                progress = step / steps_per_cycle
                angle = center_angle + swing_arc_degrees * (1 if step % 2 == 0 else -1) * (0.5 + 0.5 * progress)
                timestamp = cycle * cycle_duration + (step / steps_per_cycle) * cycle_duration
                frames.append(MotionFrame(timestamp_ms=timestamp, motor_angles={motor_id: angle}))

    else:  # SLOW_WIDE
        # Slow relaxed wag: deliberate sweeps, wider amplitude
        num_cycles = 2
        cycle_duration = duration_ms / num_cycles
        steps_per_cycle = 16

        for cycle in range(num_cycles):
            for step in range(steps_per_cycle):
                progress = step / steps_per_cycle
                # Smooth cosine wave for relaxed motion
                import math
                angle = center_angle + swing_arc_degrees * math.cos(progress * math.pi)
                timestamp = cycle * cycle_duration + (step / steps_per_cycle) * cycle_duration
                frames.append(MotionFrame(timestamp_ms=timestamp, motor_angles={motor_id: angle}))

    # Add final frame at end position
    frames.append(MotionFrame(timestamp_ms=duration_ms, motor_angles={motor_id: center_angle}))
    return frames


def head_tilt(
    tilt_angle_degrees: float = 15.0,
    duration_ms: float = 300.0,
    motor_id: str = "head",
) -> list[MotionFrame]:
    """Create a subtle diagonal head pivot for listening pose.

    Args:
        tilt_angle_degrees: Maximum tilt from vertical in degrees.
        duration_ms: Total motion duration in milliseconds.
        motor_id: Identifier for the head servo motor.

    Returns:
        List of MotionFrame objects defining the tilt path.
    """
    frames = []
    import math

    # Smooth tilt motion: center -> tilt -> center
    steps = 12
    center_angle = 0.0

    for step in range(steps + 1):
        progress = step / steps
        # Cosine motion: smooth acceleration and deceleration
        # Tilt to the right on first half, back to center on second half
        tilt_progress = math.sin(progress * math.pi)
        angle = center_angle + tilt_angle_degrees * tilt_progress
        timestamp = (progress * duration_ms)
        frames.append(MotionFrame(timestamp_ms=timestamp, motor_angles={motor_id: angle}))

    return frames


def walk_or_trot(
    gait: WalkGait,
    stride_count: int,
    stride_length_mm: float = 50.0,
    front_left_motor_id: str = "front_left",
    front_right_motor_id: str = "front_right",
    back_left_motor_id: str = "back_left",
    back_right_motor_id: str = "back_right",
) -> list[MotionFrame]:
    """Generate alternating four-legged locomotion motion.

    Args:
        gait: Walk style (steady for younger kids, trot for high-energy).
        stride_count: Number of complete stride cycles.
        stride_length_mm: Distance per stride in millimeters.
        front_left_motor_id: Identifier for front-left leg servo.
        front_right_motor_id: Identifier for front-right leg servo.
        back_left_motor_id: Identifier for back-left leg servo.
        back_right_motor_id: Identifier for back-right leg servo.

    Returns:
        List of MotionFrame objects defining the leg sequence.
    """
    frames = []
    import math

    if gait == WalkGait.STEADY_WALK:
        # Steady walk: alternating diagonal leg pairs
        # Front-left and back-right move together, then front-right and back-left
        steps_per_stride = 8
        stride_duration_ms = 500.0
        total_duration_ms = stride_duration_ms * stride_count

        for stride in range(stride_count):
            stride_start = stride * stride_duration_ms

            for step in range(steps_per_stride):
                progress = step / steps_per_stride
                timestamp = stride_start + (progress * stride_duration_ms)

                # Diagonal gait pattern
                if progress < 0.5:
                    # First half: FL and BR lift
                    lift_progress = progress * 2
                    fl_angle = stride_length_mm * 0.3 * math.sin(lift_progress * math.pi)
                    br_angle = stride_length_mm * 0.3 * math.sin(lift_progress * math.pi)
                    fr_angle = 0.0
                    bl_angle = 0.0
                else:
                    # Second half: FR and BL lift
                    lift_progress = (progress - 0.5) * 2
                    fl_angle = 0.0
                    br_angle = 0.0
                    fr_angle = stride_length_mm * 0.3 * math.sin(lift_progress * math.pi)
                    bl_angle = stride_length_mm * 0.3 * math.sin(lift_progress * math.pi)

                frames.append(MotionFrame(
                    timestamp_ms=timestamp,
                    motor_angles={
                        front_left_motor_id: fl_angle,
                        front_right_motor_id: fr_angle,
                        back_left_motor_id: bl_angle,
                        back_right_motor_id: br_angle,
                    },
                ))

    else:  # FAST_TROT
        # Fast trot: diagonal pairs move together simultaneously
        steps_per_stride = 6
        stride_duration_ms = 300.0
        total_duration_ms = stride_duration_ms * stride_count

        for stride in range(stride_count):
            stride_start = stride * stride_duration_ms

            for step in range(steps_per_stride):
                progress = step / steps_per_stride
                timestamp = stride_start + (progress * stride_duration_ms)

                # Trot pattern: diagonal pairs move together
                if progress < 0.5:
                    # FL and BR up
                    lift_progress = progress * 2
                    fl_angle = stride_length_mm * 0.4 * math.sin(lift_progress * math.pi)
                    br_angle = stride_length_mm * 0.4 * math.sin(lift_progress * math.pi)
                    fr_angle = -stride_length_mm * 0.2
                    bl_angle = -stride_length_mm * 0.2
                else:
                    # FR and BL up
                    lift_progress = (progress - 0.5) * 2
                    fl_angle = -stride_length_mm * 0.2
                    br_angle = -stride_length_mm * 0.2
                    fr_angle = stride_length_mm * 0.4 * math.sin(lift_progress * math.pi)
                    bl_angle = stride_length_mm * 0.4 * math.sin(lift_progress * math.pi)

                frames.append(MotionFrame(
                    timestamp_ms=timestamp,
                    motor_angles={
                        front_left_motor_id: fl_angle,
                        front_right_motor_id: fr_angle,
                        back_left_motor_id: bl_angle,
                        back_right_motor_id: br_angle,
                    },
                ))

    # Return to neutral position
    frames.append(MotionFrame(
        timestamp_ms=total_duration_ms,
        motor_angles={
            front_left_motor_id: 0.0,
            front_right_motor_id: 0.0,
            back_left_motor_id: 0.0,
            back_right_motor_id: 0.0,
        },
    ))

    return frames


def paws_and_begging(
    hip_hinge_angle_degrees: float = 90.0,
    front_paw_lift_height_mm: float = 30.0,
    hold_duration_ms: float = 1000.0,
    hip_motor_id: str = "hip",
    front_paw_motor_id: str = "front_paw",
) -> list[MotionFrame]:
    """Generate a sitting begging posture with raised front paws.

    Dual-axis hip hinge allows the toy to rock back on hind legs
    while lifting front paws into a begging pose.

    Args:
        hip_hinge_angle_degrees: Hip rotation angle in degrees.
        front_paw_lift_height_mm: Height of front paw lift in millimeters.
        hold_duration_ms: Duration to hold the pose in milliseconds.
        hip_motor_id: Identifier for the hip servo motor.
        front_paw_motor_id: Identifier for the front paw servo motor.

    Returns:
        List of MotionFrame objects defining the begging sequence.
    """
    frames = []
    import math

    # Phase 1: Transition to begging (0 to 500ms)
    transition_duration = 500.0
    steps_transition = 10

    for step in range(steps_transition + 1):
        progress = step / steps_transition
        timestamp = progress * transition_duration
        # Smooth transition using sine wave
        hip_angle = hip_hinge_angle_degrees * math.sin(progress * math.pi / 2)
        paw_height = front_paw_lift_height_mm * math.sin(progress * math.pi / 2)
        frames.append(MotionFrame(
            timestamp_ms=timestamp,
            motor_angles={
                hip_motor_id: hip_angle,
                front_paw_motor_id: paw_height,
            },
        ))

    # Phase 2: Hold begging pose (500ms to 500 + hold_duration_ms)
    hold_start = transition_duration
    hold_end = hold_start + hold_duration_ms
    hold_steps = 5

    for step in range(hold_steps):
        progress = step / (hold_steps - 1) if hold_steps > 1 else 0
        # Small subtle movements while holding pose (slight wobble)
        wobble = 2.0 * math.sin(progress * math.pi * 2)
        timestamp = hold_start + (progress * hold_duration_ms)
        frames.append(MotionFrame(
            timestamp_ms=timestamp,
            motor_angles={
                hip_motor_id: hip_hinge_angle_degrees + wobble,
                front_paw_motor_id: front_paw_lift_height_mm,
            },
        ))

    # Phase 3: Return to neutral (hold_end to hold_end + 500ms)
    return_duration = 500.0
    steps_return = 10

    for step in range(steps_return + 1):
        progress = step / steps_return
        timestamp = hold_end + (progress * return_duration)
        # Smooth return to neutral
        hip_angle = hip_hinge_angle_degrees * math.cos(progress * math.pi / 2)
        paw_height = front_paw_lift_height_mm * math.cos(progress * math.pi / 2)
        frames.append(MotionFrame(
            timestamp_ms=timestamp,
            motor_angles={
                hip_motor_id: hip_angle,
                front_paw_motor_id: paw_height,
            },
        ))

    return frames


def ear_twitch(
    motion_type: EarMotion,
    twitch_count: int = 3,
    frequency_hz: float = 5.0,
    left_ear_motor_id: str = "left_ear",
    right_ear_motor_id: str = "right_ear",
) -> list[MotionFrame]:
    """Generate quick ear motion (vertical or flaring) for attention response.

    Args:
        motion_type: Type of ear motion (vertical pivot or flaring spread).
        twitch_count: Number of twitch cycles.
        frequency_hz: Motion frequency in hertz.
        left_ear_motor_id: Identifier for the left ear servo motor.
        right_ear_motor_id: Identifier for the right ear servo motor.

    Returns:
        List of MotionFrame objects defining the ear motion.
    """
    frames = []
    import math

    cycle_duration_ms = 1000.0 / frequency_hz
    total_duration_ms = cycle_duration_ms * twitch_count

    if motion_type == EarMotion.VERTICAL:
        # Vertical ear motion: ears move up and down together
        for cycle in range(twitch_count):
            cycle_start = cycle * cycle_duration_ms
            steps_per_cycle = 8

            for step in range(steps_per_cycle):
                progress = step / steps_per_cycle
                timestamp = cycle_start + (progress * cycle_duration_ms)
                # Sine wave for smooth up-down motion
                ear_angle = 20.0 * math.sin(progress * math.pi * 2)
                frames.append(MotionFrame(
                    timestamp_ms=timestamp,
                    motor_angles={
                        left_ear_motor_id: ear_angle,
                        right_ear_motor_id: ear_angle,
                    },
                ))

    else:  # FLARING
        # Flaring ear motion: ears spread outward and inward
        for cycle in range(twitch_count):
            cycle_start = cycle * cycle_duration_ms
            steps_per_cycle = 8

            for step in range(steps_per_cycle):
                progress = step / steps_per_cycle
                timestamp = cycle_start + (progress * cycle_duration_ms)
                # Left ear goes left, right ear goes right
                flare_angle = 25.0 * math.sin(progress * math.pi * 2)
                frames.append(MotionFrame(
                    timestamp_ms=timestamp,
                    motor_angles={
                        left_ear_motor_id: -flare_angle,
                        right_ear_motor_id: flare_angle,
                    },
                ))

    # Return to neutral position
    frames.append(MotionFrame(
        timestamp_ms=total_duration_ms,
        motor_angles={
            left_ear_motor_id: 0.0,
            right_ear_motor_id: 0.0,
        },
    ))

    return frames


def panting_interaction(
    jaw_travel_mm: float = 15.0,
    pant_cycles: int = 5,
    cycle_duration_ms: float = 200.0,
    jaw_motor_id: str = "jaw",
) -> list[MotionFrame]:
    """Generate a rapid jaw up-and-down motion simulating happy panting.

    Args:
        jaw_travel_mm: Distance jaw travels in millimeters.
        pant_cycles: Number of pant cycles.
        cycle_duration_ms: Duration per complete cycle in milliseconds.
        jaw_motor_id: Identifier for the jaw servo motor.

    Returns:
        List of MotionFrame objects defining the panting sequence.
    """
    frames = []
    import math

    total_duration_ms = cycle_duration_ms * pant_cycles

    for cycle in range(pant_cycles):
        cycle_start = cycle * cycle_duration_ms
        steps_per_cycle = 6

        for step in range(steps_per_cycle):
            progress = step / steps_per_cycle
            timestamp = cycle_start + (progress * cycle_duration_ms)
            # Triangle wave for quick open-close motion
            # Open phase (0 to 0.5) then close phase (0.5 to 1.0)
            if progress < 0.5:
                # Opening jaw
                jaw_position = jaw_travel_mm * (progress * 2)
            else:
                # Closing jaw
                jaw_position = jaw_travel_mm * (2 - progress * 2)

            frames.append(MotionFrame(
                timestamp_ms=timestamp,
                motor_angles={jaw_motor_id: jaw_position},
            ))

    # Return to neutral (closed) position
    frames.append(MotionFrame(
        timestamp_ms=total_duration_ms,
        motor_angles={jaw_motor_id: 0.0},
    ))

    return frames


def begging(
    rear_leg_lift_height_mm: float = 40.0,
    front_paw_lift_height_mm: float = 35.0,
    hold_duration_ms: float = 1500.0,
    rear_leg_motor_id: str = "rear_legs",
    front_paw_motor_id: str = "front_paws",
) -> list[MotionFrame]:
    """Generate a begging posture with rear legs lifted and front paws raised.

    Dog rocks back on rear legs while lifting front paws in a begging pose.
    More exaggerated than paws_and_begging for dramatic effect.

    Args:
        rear_leg_lift_height_mm: Height of rear leg lift in millimeters.
        front_paw_lift_height_mm: Height of front paw lift in millimeters.
        hold_duration_ms: Duration to hold the pose in milliseconds.
        rear_leg_motor_id: Identifier for the rear leg servo motor.
        front_paw_motor_id: Identifier for the front paw servo motor.

    Returns:
        List of MotionFrame objects defining the begging sequence.
    """
    frames = []
    import math

    # Phase 1: Transition to begging (0 to 600ms)
    transition_duration = 600.0
    steps_transition = 12

    for step in range(steps_transition + 1):
        progress = step / steps_transition
        timestamp = progress * transition_duration
        # Smooth transition using sine wave
        rear_lift = rear_leg_lift_height_mm * math.sin(progress * math.pi / 2)
        paw_height = front_paw_lift_height_mm * math.sin(progress * math.pi / 2)
        frames.append(MotionFrame(
            timestamp_ms=timestamp,
            motor_angles={
                rear_leg_motor_id: rear_lift,
                front_paw_motor_id: paw_height,
            },
        ))

    # Phase 2: Hold begging pose (600ms to 600 + hold_duration_ms)
    hold_start = transition_duration
    hold_end = hold_start + hold_duration_ms
    hold_steps = 8

    for step in range(hold_steps):
        progress = step / (hold_steps - 1) if hold_steps > 1 else 0
        # Subtle rocking motion while holding pose
        rock = 3.0 * math.sin(progress * math.pi * 2)
        timestamp = hold_start + (progress * hold_duration_ms)
        frames.append(MotionFrame(
            timestamp_ms=timestamp,
            motor_angles={
                rear_leg_motor_id: rear_leg_lift_height_mm + rock,
                front_paw_motor_id: front_paw_lift_height_mm,
            },
        ))

    # Phase 3: Return to neutral (hold_end to hold_end + 600ms)
    return_duration = 600.0
    steps_return = 12

    for step in range(steps_return + 1):
        progress = step / steps_return
        timestamp = hold_end + (progress * return_duration)
        # Smooth return to neutral
        rear_lift = rear_leg_lift_height_mm * math.cos(progress * math.pi / 2)
        paw_height = front_paw_lift_height_mm * math.cos(progress * math.pi / 2)
        frames.append(MotionFrame(
            timestamp_ms=timestamp,
            motor_angles={
                rear_leg_motor_id: rear_lift,
                front_paw_motor_id: paw_height,
            },
        ))

    return frames


def sit(
    spine_angle_degrees: float = 90.0,
    head_nod_angle_degrees: float = 15.0,
    sit_duration_ms: float = 2000.0,
    spine_motor_id: str = "spine",
    head_motor_id: str = "head",
) -> list[MotionFrame]:
    """Generate a sitting posture with optional head nod.

    Dog transitions to a sitting position with spine bent at specified angle.
    Can optionally include a subtle head nod while sitting.

    Args:
        spine_angle_degrees: Bend angle for sitting posture in degrees.
        head_nod_angle_degrees: Subtle head nod angle in degrees.
        sit_duration_ms: Duration to hold the sitting pose in milliseconds.
        spine_motor_id: Identifier for the spine servo motor.
        head_motor_id: Identifier for the head servo motor.

    Returns:
        List of MotionFrame objects defining the sitting sequence.
    """
    frames = []
    import math

    # Phase 1: Transition to sitting (0 to 800ms)
    transition_duration = 800.0
    steps_transition = 16

    for step in range(steps_transition + 1):
        progress = step / steps_transition
        timestamp = progress * transition_duration
        # Smooth transition using sine wave
        spine_bend = spine_angle_degrees * math.sin(progress * math.pi / 2)
        head_angle = 0.0  # Head stays neutral during sit
        frames.append(MotionFrame(
            timestamp_ms=timestamp,
            motor_angles={
                spine_motor_id: spine_bend,
                head_motor_id: head_angle,
            },
        ))

    # Phase 2: Hold sitting pose with head nod (800ms to 800 + sit_duration_ms)
    sit_start = transition_duration
    sit_end = sit_start + sit_duration_ms
    sit_steps = 12

    for step in range(sit_steps):
        progress = step / (sit_steps - 1) if sit_steps > 1 else 0
        # Gentle head nod while sitting
        head_nod = head_nod_angle_degrees * math.sin(progress * math.pi * 2)
        timestamp = sit_start + (progress * sit_duration_ms)
        frames.append(MotionFrame(
            timestamp_ms=timestamp,
            motor_angles={
                spine_motor_id: spine_angle_degrees,
                head_motor_id: head_nod,
            },
        ))

    # Phase 3: Return to neutral standing (sit_end to sit_end + 800ms)
    return_duration = 800.0
    steps_return = 16

    for step in range(steps_return + 1):
        progress = step / steps_return
        timestamp = sit_end + (progress * return_duration)
        # Smooth return to neutral
        spine_bend = spine_angle_degrees * math.cos(progress * math.pi / 2)
        head_angle = 0.0
        frames.append(MotionFrame(
            timestamp_ms=timestamp,
            motor_angles={
                spine_motor_id: spine_bend,
                head_motor_id: head_angle,
            },
        ))

    return frames
