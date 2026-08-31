import random
from typing import Optional
from dog_scheduler import DogState, DogStateMetrics
from dogmove import (
    MotionFrame,
    tail_wag,
    head_tilt,
    walk_or_trot,
    paws_and_begging,
    ear_twitch,
    panting_interaction,
    WagStyle,
    WalkGait,
    EarMotion,
)


class BehaviorOrchestrator:
    """Maps high-level dog states to movement sequences."""

    def __init__(self, use_variation: bool = True):
        """
        Initialize orchestrator.

        Args:
            use_variation: Add randomization to movements for naturalness
        """
        self.use_variation = use_variation
        self.last_behavior: Optional[DogState] = None

    def get_movements_for_state(
        self, dog_state: DogState, intensity: float = 0.7, duration: int = 3000
    ) -> list[MotionFrame]:
        """
        Get motion frames for a given dog state.

        Args:
            dog_state: Current state (SLEEPING, ALERT, PLAYING, etc.)
            intensity: 0.0-1.0 intensity multiplier
            duration: Desired duration in milliseconds

        Returns:
            List of MotionFrame objects
        """
        if dog_state == DogState.SLEEPING:
            movements = self._sleeping_movements(intensity, duration)
        elif dog_state == DogState.ALERT:
            movements = self._alert_movements(intensity, duration)
        elif dog_state == DogState.PLAYING:
            movements = self._playing_movements(intensity, duration)
        elif dog_state == DogState.RESTING:
            movements = self._resting_movements(intensity, duration)
        elif dog_state == DogState.EXPLORING:
            movements = self._exploring_movements(intensity, duration)
        elif dog_state == DogState.EATING:
            movements = self._eating_movements(intensity, duration)
        elif dog_state == DogState.SCRATCHING:
            movements = self._scratching_movements(intensity, duration)
        elif dog_state == DogState.GROOMING:
            movements = self._grooming_movements(intensity, duration)
        else:
            movements = self._idle_movements(duration)

        return self.add_idle_variation(movements) if self.use_variation else movements

    def _sleeping_movements(self, intensity: float, duration: int) -> list[MotionFrame]:
        """Sleeping: occasional ear twitches, position shifts, REM jerks."""
        movements = []
        elapsed = 0

        while elapsed < duration:
            # Occasional ear twitch (dreaming)
            if random.random() < 0.3:
                twitch = ear_twitch(
                    motion_type=EarMotion.VERTICAL if random.random() < 0.5 else EarMotion.FLARING,
                    twitch_count=random.randint(1, 2),
                    frequency_hz=random.uniform(2, 4),
                )
                movements.extend(twitch)
                elapsed += twitch[-1].timestamp_ms if twitch else 0

            # Position shift
            if random.random() < 0.4:
                shift_frames = [
                    MotionFrame(timestamp_ms=elapsed, motor_angles={"body": random.uniform(-5, 5)}),
                    MotionFrame(timestamp_ms=elapsed + 500, motor_angles={"body": 0}),
                ]
                movements.extend(shift_frames)
                elapsed += 500

            # Random pause
            pause_duration = random.randint(2000, 5000)
            movements.append(MotionFrame(timestamp_ms=elapsed, motor_angles={}))
            elapsed += pause_duration

            if elapsed >= duration:
                break

        return self._normalize_timestamps(movements[:100], duration)

    def _alert_movements(self, intensity: float, duration: int) -> list[MotionFrame]:
        """Alert: head tilts, ear flares, occasional scans."""
        movements = []
        elapsed = 0

        while elapsed < duration:
            # Head tilt (looking around)
            if random.random() < 0.6:
                angle = random.choice([-15, 15])
                tilt = head_tilt(tilt_angle_degrees=angle, duration_ms=300)
                movements.extend(tilt)
                elapsed += 300

            # Ear flare
            if random.random() < 0.5:
                flare = ear_twitch(motion_type=EarMotion.FLARING, twitch_count=2, frequency_hz=random.uniform(2, 4))
                movements.extend(flare)
                elapsed += flare[-1].timestamp_ms if flare else 0

            # Pause and listen
            pause_duration = random.randint(500, 2000)
            movements.append(MotionFrame(timestamp_ms=elapsed, motor_angles={}))
            elapsed += pause_duration

            if elapsed >= duration:
                break

        return self._normalize_timestamps(movements[:50], duration)

    def _playing_movements(self, intensity: float, duration: int) -> list[MotionFrame]:
        """Playing: tail wags, walks/trots, paws up, begging poses."""
        movements = []
        elapsed = 0

        while elapsed < duration:
            choice = random.random()

            if choice < 0.3:
                # Excited tail wag
                wag = tail_wag(
                    style=WagStyle.FAST_LOOSE,
                    duration_ms=int(500 * intensity),
                    swing_arc_degrees=45,
                )
                movements.extend(wag)
                elapsed += wag[-1].timestamp_ms if wag else 0

            elif choice < 0.6:
                # Trot around
                walk = walk_or_trot(gait=WalkGait.FAST_TROT, stride_count=random.randint(2, 4), stride_length_mm=60)
                movements.extend(walk)
                elapsed += walk[-1].timestamp_ms if walk else 0

            elif choice < 0.8:
                # Begging/play bow
                bow = paws_and_begging(
                    hip_hinge_angle_degrees=random.uniform(30, 50),
                    front_paw_lift_height_mm=20,
                    hold_duration_ms=500,
                )
                movements.extend(bow)
                elapsed += bow[-1].timestamp_ms if bow else 0

            else:
                # Head tilt playfully
                tilt = head_tilt(tilt_angle_degrees=random.choice([-20, 20]), duration_ms=200)
                movements.extend(tilt)
                elapsed += 200

            if elapsed >= duration:
                break

        return self._normalize_timestamps(movements[:150], duration)

    def _resting_movements(self, intensity: float, duration: int) -> list[MotionFrame]:
        """Resting: slow breathing, occasional repositioning, head droop."""
        movements = []
        elapsed = 0

        while elapsed < duration:
            # Slow breathing (subtle chest/body movement)
            if random.random() < 0.7:
                breath_frames = [
                    MotionFrame(timestamp_ms=elapsed, motor_angles={"chest": 2}),
                    MotionFrame(timestamp_ms=elapsed + 1000, motor_angles={"chest": 0}),
                    MotionFrame(timestamp_ms=elapsed + 2000, motor_angles={"chest": -2}),
                    MotionFrame(timestamp_ms=elapsed + 3000, motor_angles={"chest": 0}),
                ]
                movements.extend(breath_frames)
                elapsed += 3000

            # Occasional position shift
            if random.random() < 0.4:
                shift = head_tilt(tilt_angle_degrees=random.uniform(-5, 5), duration_ms=300)
                movements.extend(shift)
                elapsed += 300

            # Long pause
            pause_duration = random.randint(2000, 5000)
            movements.append(MotionFrame(timestamp_ms=elapsed, motor_angles={}))
            elapsed += pause_duration

            if elapsed >= duration:
                break

        return self._normalize_timestamps(movements[:80], duration)

    def _exploring_movements(self, intensity: float, duration: int) -> list[MotionFrame]:
        """Exploring: varied walks, directional head turns, sniffing."""
        movements = []
        elapsed = 0

        while elapsed < duration:
            choice = random.random()

            if choice < 0.4:
                # Sniffing (head down and side sweeps)
                sniff_frames = [
                    MotionFrame(timestamp_ms=elapsed, motor_angles={"head": -30}),
                    MotionFrame(timestamp_ms=elapsed + 300, motor_angles={"head": 0}),
                    MotionFrame(timestamp_ms=elapsed + 600, motor_angles={"head": 30}),
                    MotionFrame(timestamp_ms=elapsed + 900, motor_angles={"head": 0}),
                ]
                movements.extend(sniff_frames)
                elapsed += 900

            elif choice < 0.7:
                # Steady walk
                walk = walk_or_trot(gait=WalkGait.STEADY_WALK, stride_count=random.randint(2, 3), stride_length_mm=50)
                movements.extend(walk)
                elapsed += walk[-1].timestamp_ms if walk else 0

            else:
                # Scanning look (head tilts in different directions)
                scan_angle = random.choice([-25, -15, 0, 15, 25])
                scan = head_tilt(tilt_angle_degrees=scan_angle, duration_ms=250)
                movements.extend(scan)
                elapsed += 250

            if elapsed >= duration:
                break

        return self._normalize_timestamps(movements[:100], duration)

    def _eating_movements(self, intensity: float, duration: int) -> list[MotionFrame]:
        """Eating: head down, jaw movements, occasional head raises."""
        movements = []
        elapsed = 0

        while elapsed < duration:
            # Head down eating posture
            head_down_frames = [
                MotionFrame(timestamp_ms=elapsed, motor_angles={"head": -45}),
                MotionFrame(timestamp_ms=elapsed + 200, motor_angles={"head": -45}),
            ]
            movements.extend(head_down_frames)
            elapsed += 200

            # Eating motion (jaw movement)
            if random.random() < 0.8:
                eat = panting_interaction(
                    jaw_travel_mm=20, pant_cycles=random.randint(3, 6), cycle_duration_ms=300
                )
                movements.extend(eat)
                elapsed += eat[-1].timestamp_ms if eat else 0

            # Occasional head raise to check surroundings
            if random.random() < 0.3:
                head_up = head_tilt(tilt_angle_degrees=0, duration_ms=200)
                movements.extend(head_up)
                elapsed += 200

            # Short pause
            pause_duration = random.randint(500, 1500)
            movements.append(MotionFrame(timestamp_ms=elapsed, motor_angles={}))
            elapsed += pause_duration

            if elapsed >= duration:
                break

        return self._normalize_timestamps(movements[:100], duration)

    def _scratching_movements(self, intensity: float, duration: int) -> list[MotionFrame]:
        """Scratching: body contortions, leg movements, head turns."""
        movements = []
        elapsed = 0

        while elapsed < duration:
            choice = random.random()

            if choice < 0.4:
                # Rear leg scratch motion
                scratch_frames = [
                    MotionFrame(timestamp_ms=elapsed, motor_angles={"rear_legs": 0}),
                    MotionFrame(timestamp_ms=elapsed + 150, motor_angles={"rear_legs": 30}),
                    MotionFrame(timestamp_ms=elapsed + 300, motor_angles={"rear_legs": 0}),
                ]
                movements.extend(scratch_frames)
                elapsed += 300

            elif choice < 0.7:
                # Body twist/contortion
                twist_frames = [
                    MotionFrame(timestamp_ms=elapsed, motor_angles={"spine": 0}),
                    MotionFrame(timestamp_ms=elapsed + 200, motor_angles={"spine": 20}),
                    MotionFrame(timestamp_ms=elapsed + 400, motor_angles={"spine": -20}),
                    MotionFrame(timestamp_ms=elapsed + 600, motor_angles={"spine": 0}),
                ]
                movements.extend(twist_frames)
                elapsed += 600

            else:
                # Head turn to affected area
                head_turn = head_tilt(tilt_angle_degrees=random.choice([-30, 30]), duration_ms=300)
                movements.extend(head_turn)
                elapsed += 300

            if elapsed >= duration:
                break

        return self._normalize_timestamps(movements[:80], duration)

    def _grooming_movements(self, intensity: float, duration: int) -> list[MotionFrame]:
        """Grooming: licking simulation, ear cleaning, body straightening."""
        movements = []
        elapsed = 0

        while elapsed < duration:
            choice = random.random()

            if choice < 0.4:
                # Lick simulation (head down, jaw movements)
                lick_frames = [
                    MotionFrame(timestamp_ms=elapsed, motor_angles={"head": -20}),
                    MotionFrame(timestamp_ms=elapsed + 200, motor_angles={"head": -20}),
                ]
                movements.extend(lick_frames)
                elapsed += 200

                # Jaw motion
                lick = panting_interaction(jaw_travel_mm=15, pant_cycles=3, cycle_duration_ms=200)
                movements.extend(lick)
                elapsed += lick[-1].timestamp_ms if lick else 0

            elif choice < 0.7:
                # Ear cleaning (ear back, head turn)
                ear_clean = ear_twitch(motion_type=EarMotion.VERTICAL, twitch_count=3, frequency_hz=3)
                movements.extend(ear_clean)
                elapsed += ear_clean[-1].timestamp_ms if ear_clean else 0

            else:
                # Body straightening (stretch-like motion)
                stretch_frames = [
                    MotionFrame(timestamp_ms=elapsed, motor_angles={"spine": 0}),
                    MotionFrame(timestamp_ms=elapsed + 300, motor_angles={"spine": 10}),
                    MotionFrame(timestamp_ms=elapsed + 600, motor_angles={"spine": 0}),
                ]
                movements.extend(stretch_frames)
                elapsed += 600

            if elapsed >= duration:
                break

        return self._normalize_timestamps(movements[:100], duration)

    def _idle_movements(self, duration: int) -> list[MotionFrame]:
        """Fallback: simple neutral posture."""
        return [MotionFrame(timestamp_ms=0, motor_angles={})]

    def add_idle_variation(self, base_movements: list[MotionFrame]) -> list[MotionFrame]:
        """Add slight randomization to movements for naturalness."""
        if not base_movements:
            return base_movements

        varied = []
        for frame in base_movements:
            # Add ±10% timing jitter
            jitter = random.uniform(0.9, 1.1)
            new_timestamp = frame.timestamp_ms * jitter

            # Add ±5% angle variation
            new_angles = {}
            for motor_id, angle in frame.motor_angles.items():
                angle_jitter = random.uniform(0.95, 1.05)
                new_angles[motor_id] = angle * angle_jitter

            varied.append(MotionFrame(timestamp_ms=new_timestamp, motor_angles=new_angles))

        return varied

    def create_transition(
        self, from_state: DogState, to_state: DogState, duration: int = 300
    ) -> list[MotionFrame]:
        """
        Create smooth transition between states.

        Args:
            from_state: Starting state
            to_state: Target state
            duration: Transition duration in ms

        Returns:
            Transition motion frames
        """
        if from_state == to_state:
            return []

        # Simple transition: head tilt or neutral
        if to_state == DogState.SLEEPING:
            return [MotionFrame(timestamp_ms=0, motor_angles={"head": -15})]
        elif to_state == DogState.ALERT:
            return head_tilt(tilt_angle_degrees=0, duration_ms=200)
        elif to_state == DogState.PLAYING:
            return [MotionFrame(timestamp_ms=0, motor_angles={"tail": 10})]

        return [MotionFrame(timestamp_ms=0, motor_angles={})]

    def _normalize_timestamps(
        self, frames: list[MotionFrame], target_duration: int
    ) -> list[MotionFrame]:
        """
        Normalize frame timestamps to fit target duration.

        Args:
            frames: Raw motion frames
            target_duration: Desired total duration in ms

        Returns:
            Normalized frames
        """
        if not frames:
            return frames

        # Get current total duration
        max_timestamp = max((f.timestamp_ms for f in frames), default=0)

        if max_timestamp == 0:
            return frames

        # Scale all timestamps
        scale_factor = target_duration / max_timestamp
        normalized = []

        for frame in frames:
            new_timestamp = frame.timestamp_ms * scale_factor
            # Clamp to target duration
            new_timestamp = min(new_timestamp, target_duration)
            normalized.append(
                MotionFrame(timestamp_ms=new_timestamp, motor_angles=frame.motor_angles)
            )

        return normalized

    def compose_behaviors(
        self, behavior_sequence: list[tuple[DogState, int]]
    ) -> list[MotionFrame]:
        """
        Compose multiple behaviors into one sequence.

        Args:
            behavior_sequence: List of (DogState, duration_ms) tuples

        Returns:
            Combined motion frames with proper timing
        """
        all_movements = []
        current_timestamp = 0

        for dog_state, duration in behavior_sequence:
            # Get movements for this behavior
            movements = self.get_movements_for_state(dog_state, intensity=0.7, duration=duration)

            # Shift timestamps to fit after previous behavior
            offset_movements = []
            for frame in movements:
                new_frame = MotionFrame(
                    timestamp_ms=frame.timestamp_ms + current_timestamp,
                    motor_angles=frame.motor_angles,
                )
                offset_movements.append(new_frame)

            all_movements.extend(offset_movements)
            current_timestamp += duration

        return all_movements
