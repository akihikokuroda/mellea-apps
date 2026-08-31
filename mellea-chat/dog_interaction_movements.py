"""Generate synchronized movement patterns for dog-to-dog interactions."""

import random
import math
from dogmove import MotionFrame, WagStyle, WalkGait, EarMotion
from dogmove import tail_wag, head_tilt, walk_or_trot, paws_and_begging, ear_twitch


def chase_sequence(
    pursuer_id: str,
    runner_id: str,
    duration_ms: int,
    intensity: float = 0.8,
) -> list[MotionFrame]:
    """
    Generate chase interaction movements for two dogs.

    Pursuer: fast trotting, forward-focused head
    Runner: faster walk/trot, frequent direction changes

    Args:
        pursuer_id: ID of dog doing the chasing
        runner_id: ID of dog being chased
        duration_ms: Total interaction duration in milliseconds
        intensity: Movement intensity (0.0-1.0)

    Returns:
        List of MotionFrame objects with prefixed motor IDs
    """
    movements = []
    elapsed = 0
    half_duration = duration_ms / 2

    while elapsed < duration_ms:
        # Pursuer: fast forward motion
        if elapsed < duration_ms * 0.8:
            # Fast trot sequences
            trot_frames = walk_or_trot(
                gait=WalkGait.FAST_TROT,
                stride_count=random.randint(2, 3),
                stride_length_mm=int(60 * intensity),
            )

            for frame in trot_frames:
                new_timestamp = elapsed + frame.timestamp_ms * (half_duration / (trot_frames[-1].timestamp_ms or 1000))
                if new_timestamp < duration_ms:
                    # Prefix motors for pursuer
                    prefixed = {f"{pursuer_id}:{k}": v for k, v in frame.motor_angles.items()}
                    movements.append(MotionFrame(timestamp_ms=new_timestamp, motor_angles=prefixed))

            elapsed += half_duration * 0.4

        # Runner: occasional evasion
        if elapsed < duration_ms * 0.8:
            # Direction change (head turn and slight body shift)
            evasion = head_tilt(tilt_angle_degrees=random.choice([-30, 30]), duration_ms=300)
            for frame in evasion:
                new_timestamp = elapsed + frame.timestamp_ms
                if new_timestamp < duration_ms:
                    prefixed = {f"{runner_id}:{k}": v for k, v in frame.motor_angles.items()}
                    movements.append(MotionFrame(timestamp_ms=new_timestamp, motor_angles=prefixed))

            elapsed += 300

        # Both dogs brief synchronization (runner speeds up)
        if random.random() < 0.4 and elapsed < duration_ms * 0.6:
            trot_frames = walk_or_trot(
                gait=WalkGait.FAST_TROT,
                stride_count=random.randint(1, 2),
                stride_length_mm=int(50 * intensity),
            )

            for frame in trot_frames:
                new_timestamp = elapsed + frame.timestamp_ms
                if new_timestamp < duration_ms:
                    prefixed_runner = {f"{runner_id}:{k}": v for k, v in frame.motor_angles.items()}
                    movements.append(MotionFrame(timestamp_ms=new_timestamp, motor_angles=prefixed_runner))

            elapsed += (trot_frames[-1].timestamp_ms or 500)

    return movements


def play_together_sequence(
    dog_1_id: str,
    dog_2_id: str,
    duration_ms: int,
    intensity: float = 0.7,
) -> list[MotionFrame]:
    """
    Generate mutual play interaction movements for two dogs.

    Both dogs: play bows, circling, synchronized tail wags, mirrored head tilts

    Args:
        dog_1_id: ID of first dog
        dog_2_id: ID of second dog
        duration_ms: Total interaction duration in milliseconds
        intensity: Movement intensity (0.0-1.0)

    Returns:
        List of MotionFrame objects with prefixed motor IDs
    """
    movements = []
    elapsed = 0

    while elapsed < duration_ms:
        choice = random.random()

        if choice < 0.3:
            # Synchronized play bows
            bow_frames = paws_and_begging(
                hip_hinge_angle_degrees=random.uniform(30, 50),
                front_paw_lift_height_mm=20,
                hold_duration_ms=500,
            )

            for frame in bow_frames:
                new_timestamp = elapsed + frame.timestamp_ms
                if new_timestamp < duration_ms:
                    # Both dogs do play bow at same time
                    prefixed_1 = {f"{dog_1_id}:{k}": v for k, v in frame.motor_angles.items()}
                    prefixed_2 = {f"{dog_2_id}:{k}": v for k, v in frame.motor_angles.items()}
                    # Merge both dogs' movements
                    combined = {**prefixed_1, **prefixed_2}
                    movements.append(MotionFrame(timestamp_ms=new_timestamp, motor_angles=combined))

            elapsed += bow_frames[-1].timestamp_ms if bow_frames else 500

        elif choice < 0.6:
            # Circling behavior: one dog moves while other mirrors
            for dog_id, direction_mult in [(dog_1_id, 1), (dog_2_id, -1)]:
                head_move = head_tilt(
                    tilt_angle_degrees=random.uniform(-20, 20) * direction_mult,
                    duration_ms=400,
                )
                for frame in head_move:
                    new_timestamp = elapsed + frame.timestamp_ms * direction_mult
                    if new_timestamp < duration_ms:
                        prefixed = {f"{dog_id}:{k}": v for k, v in frame.motor_angles.items()}
                        movements.append(MotionFrame(timestamp_ms=new_timestamp, motor_angles=prefixed))

            elapsed += 400

        else:
            # Synchronized excited tail wags
            wag_frames = tail_wag(
                style=WagStyle.FAST_LOOSE,
                duration_ms=int(600 * intensity),
                swing_arc_degrees=45,
            )

            for frame in wag_frames:
                new_timestamp = elapsed + frame.timestamp_ms
                if new_timestamp < duration_ms:
                    prefixed_1 = {f"{dog_1_id}:{k}": v for k, v in frame.motor_angles.items()}
                    prefixed_2 = {f"{dog_2_id}:{k}": v for k, v in frame.motor_angles.items()}
                    combined = {**prefixed_1, **prefixed_2}
                    movements.append(MotionFrame(timestamp_ms=new_timestamp, motor_angles=combined))

            elapsed += wag_frames[-1].timestamp_ms if wag_frames else 600

    return movements[:200]  # Cap at 200 frames


def interacting_sequence(
    dog_1_id: str,
    dog_2_id: str,
    duration_ms: int,
    interaction_type: str = "generic",
) -> list[MotionFrame]:
    """
    Generate gentle interaction movements for two dogs.

    Sniffing, circling, social grooming gestures, mutual ear attention

    Args:
        dog_1_id: ID of first dog
        dog_2_id: ID of second dog
        duration_ms: Total interaction duration in milliseconds
        interaction_type: Type of interaction

    Returns:
        List of MotionFrame objects with prefixed motor IDs
    """
    movements = []
    elapsed = 0

    while elapsed < duration_ms:
        choice = random.random()

        if choice < 0.4:
            # Mutual sniffing: heads tilted toward each other
            # Dog 1 tilts right, Dog 2 tilts left (toward each other)
            for dog_id, tilt_angle in [(dog_1_id, 15), (dog_2_id, -15)]:
                sniff_frames = head_tilt(tilt_angle_degrees=tilt_angle, duration_ms=300)
                for frame in sniff_frames:
                    new_timestamp = elapsed + frame.timestamp_ms
                    if new_timestamp < duration_ms:
                        prefixed = {f"{dog_id}:{k}": v for k, v in frame.motor_angles.items()}
                        movements.append(MotionFrame(timestamp_ms=new_timestamp, motor_angles=prefixed))

            elapsed += 300

        elif choice < 0.7:
            # Gentle circling: slow synchronized movement
            for _ in range(2):
                circle_frames = walk_or_trot(
                    gait=WalkGait.STEADY_WALK,
                    stride_count=1,
                    stride_length_mm=30,
                )
                for frame in circle_frames:
                    new_timestamp = elapsed + frame.timestamp_ms
                    if new_timestamp < duration_ms:
                        prefixed_1 = {f"{dog_1_id}:{k}": v for k, v in frame.motor_angles.items()}
                        prefixed_2 = {f"{dog_2_id}:{k}": v for k, v in frame.motor_angles.items()}
                        combined = {**prefixed_1, **prefixed_2}
                        movements.append(MotionFrame(timestamp_ms=new_timestamp, motor_angles=combined))

                elapsed += circle_frames[-1].timestamp_ms if circle_frames else 500

        else:
            # Mutual grooming gesture: slow ear twitches
            twitch_frames = ear_twitch(
                motion_type=EarMotion.VERTICAL,
                twitch_count=2,
                frequency_hz=1.5,
            )
            for frame in twitch_frames:
                new_timestamp = elapsed + frame.timestamp_ms
                if new_timestamp < duration_ms:
                    prefixed_1 = {f"{dog_1_id}:{k}": v for k, v in frame.motor_angles.items()}
                    prefixed_2 = {f"{dog_2_id}:{k}": v for k, v in frame.motor_angles.items()}
                    combined = {**prefixed_1, **prefixed_2}
                    movements.append(MotionFrame(timestamp_ms=new_timestamp, motor_angles=combined))

            elapsed += twitch_frames[-1].timestamp_ms if twitch_frames else 500

    return movements[:150]  # Cap at 150 frames
