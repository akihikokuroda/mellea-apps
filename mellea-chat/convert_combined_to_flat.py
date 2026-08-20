#!/usr/bin/env python3
"""Convert combined animation format to flat array format with sequential timing.

Converts all_movements_combined.json (with session_info and movements array)
to a flat array of motion frames suitable for the HTML animator.

Adjusts timestamps so movements play sequentially instead of simultaneously.

Usage:
    python3 convert_combined_to_flat.py
    python3 convert_combined_to_flat.py dog_animations/all_movements_combined.json
    python3 convert_combined_to_flat.py -i input.json -o output.json
    python3 convert_combined_to_flat.py --no-adjust  (don't adjust timing)
"""

import json
import argparse
from pathlib import Path


def detect_and_adjust_overlapping_movements(flat_array, adjust_timing=True, gap_ms=100):
    """Detect overlapping movements in flat array and adjust timing.

    Movements overlap when timestamps restart at 0. This adjusts them
    to play sequentially.

    Args:
        flat_array: List of motion frames
        adjust_timing: Whether to adjust timestamps
        gap_ms: Gap in milliseconds between movements

    Returns:
        Adjusted flat array with sequential timing
    """
    if not flat_array or not adjust_timing:
        return flat_array

    # Detect movement boundaries (where timestamp resets to 0 or nearby)
    movements = []
    current_movement = [flat_array[0]]
    movement_count = 1

    for i in range(1, len(flat_array)):
        frame = flat_array[i]
        prev_frame = flat_array[i - 1]

        # Check if this is a new movement (timestamp resets)
        # or motor changed significantly
        if frame['timestamp_ms'] <= prev_frame['timestamp_ms'] * 0.1:  # 10% threshold
            # New movement detected
            movements.append(current_movement)
            current_movement = [frame]
            movement_count += 1
        else:
            current_movement.append(frame)

    if current_movement:
        movements.append(current_movement)

    if movement_count > 1:
        print(f"✓ Detected {movement_count} overlapping movements in flat array")
        # Re-timestamp for sequential playback
        flat_array = []
        time_offset = 0

        for i, movement in enumerate(movements):
            if len(movement) == 0:
                continue

            movement_duration = movement[-1]['timestamp_ms']

            for frame in movement:
                new_frame = frame.copy()
                new_frame['timestamp_ms'] = frame['timestamp_ms'] + time_offset
                flat_array.append(new_frame)

            time_offset += movement_duration + gap_ms

        print(f"  - Total movements: {movement_count}")
        print(f"  - Total frames: {len(flat_array)}")
        print(f"  - Total duration: {time_offset}ms ({time_offset/1000:.1f}s)")
        print(f"  - Gap between movements: {gap_ms}ms")

    return flat_array


def flatten_combined_format(combined_data, adjust_timing=True, gap_ms=100):
    """Convert combined format or overlapping flat array to sequential format.

    Args:
        combined_data: Dict with 'session_info' and 'movements' keys, OR
                      a flat array with overlapping movements
        adjust_timing: Whether to adjust timestamps for sequential playback
        gap_ms: Gap in milliseconds between movements

    Returns:
        List of motion frames (flat array)
    """
    # Check if it's already a flat array
    if isinstance(combined_data, list):
        print("✓ Input is already a flat array")
        if adjust_timing:
            return detect_and_adjust_overlapping_movements(
                combined_data, adjust_timing=True, gap_ms=gap_ms
            )
        else:
            return combined_data

    if not isinstance(combined_data, dict):
        raise ValueError("Input must be a list (flat array) or a dictionary with 'movements' key")

    if 'movements' not in combined_data:
        raise ValueError("Input dictionary must have a 'movements' key")

    movements = combined_data['movements']

    if not isinstance(movements, list):
        raise ValueError("'movements' must be an array")

    if len(movements) == 0:
        raise ValueError("'movements' array is empty")

    # Check if movements is array of arrays (combined format)
    first_element = movements[0]
    if isinstance(first_element, dict) and 'timestamp_ms' in first_element:
        print("✓ Input is already a flat array (dict with 'movements' key)")
        if adjust_timing:
            return detect_and_adjust_overlapping_movements(
                movements, adjust_timing=True, gap_ms=gap_ms
            )
        else:
            return movements

    # Flatten all movements into one array with sequential timing
    flat_array = []
    time_offset = 0

    for i, movement in enumerate(movements):
        if not isinstance(movement, list):
            raise ValueError(f"Movement {i} is not an array")

        if len(movement) == 0:
            continue

        # Get duration of this movement
        movement_duration = movement[-1]['timestamp_ms']

        for frame in movement:
            new_frame = frame.copy()

            if adjust_timing:
                # Adjust timestamp to add offset
                new_frame['timestamp_ms'] = frame['timestamp_ms'] + time_offset
            else:
                new_frame['timestamp_ms'] = frame['timestamp_ms']

            flat_array.append(new_frame)

        # Add offset for next movement
        if adjust_timing:
            time_offset += movement_duration + gap_ms

    if adjust_timing:
        print(f"✓ Converted {len(movements)} movements with sequential timing")
        print(f"  - Total movements: {len(movements)}")
        print(f"  - Total frames: {len(flat_array)}")
        print(f"  - Total duration: {time_offset}ms ({time_offset/1000:.1f}s)")
        print(f"  - Gap between movements: {gap_ms}ms")
    else:
        print(f"✓ Converted {len(movements)} movements into flat array with {len(flat_array)} frames")

    return flat_array


def main():
    parser = argparse.ArgumentParser(
        description="Convert combined animation format to flat array format with sequential timing"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="dog_animations/all_movements_combined.json",
        help="Input JSON file (default: dog_animations/all_movements_combined.json)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file (default: overwrite input file)"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of original file"
    )
    parser.add_argument(
        "--no-adjust",
        action="store_true",
        help="Don't adjust timing (movements play simultaneously)"
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=100,
        help="Gap in milliseconds between movements (default: 100ms)"
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)

    # Check if input file exists
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return False

    print(f"📖 Reading: {input_path}")

    # Read input file
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

    # Convert to flat array
    try:
        flat_array = flatten_combined_format(
            data,
            adjust_timing=not args.no_adjust,
            gap_ms=args.gap
        )
    except ValueError as e:
        print(f"❌ Conversion error: {e}")
        return False

    # Determine output file
    output_path = Path(args.output) if args.output else input_path

    # Create backup if requested
    if args.backup and output_path.exists():
        backup_path = output_path.with_suffix('.backup.json')
        print(f"💾 Creating backup: {backup_path}")
        with open(backup_path, 'w') as f:
            json.dump(data, f, indent=2)

    # Write output file
    print(f"✍️  Writing: {output_path}")
    try:
        with open(output_path, 'w') as f:
            json.dump(flat_array, f, indent=2)
    except Exception as e:
        print(f"❌ Error writing file: {e}")
        return False

    print(f"✅ Conversion complete!")
    print(f"\n📊 Statistics:")
    print(f"  - Total frames: {len(flat_array)}")
    print(f"  - File size: {output_path.stat().st_size / 1024:.1f} KB")
    if not args.no_adjust:
        print(f"  - Timing: Sequential (movements play one after another)")
        print(f"  - Gap between movements: {args.gap}ms")
    else:
        print(f"  - Timing: Not adjusted (movements play simultaneously)")

    print(f"\n🎬 To use in animator:")
    print(f"  1. Open dog_animator.html in browser")
    print(f"  2. Copy content from {output_path}")
    print(f"  3. Paste into 'Motion Frames (JSON)' textarea")
    print(f"  4. Click '▶ Play'")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
