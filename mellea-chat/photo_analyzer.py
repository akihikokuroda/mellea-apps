#!/usr/bin/env python3
"""Analyze photo contents using Granite Vision model.

Usage:
    uv run python photo_analyzer.py <image_path> [analysis_type]

Example:
    uv run python photo_analyzer.py photo.jpg
    uv run python photo_analyzer.py photo.jpg objects
    uv run python photo_analyzer.py photo.jpg scene
"""

import sys
import json
import re
from pathlib import Path
from pydantic import BaseModel, Field
from PIL import Image

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

from mellea import start_session
from mellea.core import ImageBlock
from mellea.backends.model_options import ModelOption


def open_image(image_path: Path) -> Image.Image:
    """Open image file, with support for HEIC/HEIF formats.

    Args:
        image_path: Path to image file

    Returns:
        PIL Image object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is not supported
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    suffix = image_path.suffix.lower()

    # Check if HEIC format
    if suffix in ['.heic', '.heif']:
        if not HEIC_SUPPORT:
            raise ValueError(
                f"HEIC format not supported. Install pillow-heif: pip install pillow-heif"
            )

    try:
        img = Image.open(image_path)
        img.load()  # Force load to catch format errors early
        return img
    except Exception as e:
        raise ValueError(f"Failed to open image {image_path}: {e}")


def fix_and_parse_json(json_str: str) -> dict:
    """Attempt to fix and parse malformed JSON from model response."""
    # First try direct parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Fix common issues: extra quotes before strings in arrays
    # Replace ["" with [" and ""] with "]
    json_str = re.sub(r'\["+"', '["', json_str)
    json_str = re.sub(r'""\]', '"]', json_str)
    json_str = re.sub(r'""+,', '",', json_str)
    json_str = re.sub(r', ""+', ', "', json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Try to truncate at reasonable point and close JSON
    # Find the last complete structure
    last_bracket = json_str.rfind('}')
    last_bracket_arr = json_str.rfind(']')

    # Work backwards to find valid JSON
    for end_pos in range(len(json_str) - 1, max(len(json_str) - 500, 0), -1):
        test_str = json_str[:end_pos]

        # Try to close any open structures
        open_braces = test_str.count('{') - test_str.count('}')
        open_brackets = test_str.count('[') - test_str.count(']')
        open_quotes = (len(test_str) - len(test_str.replace('"', ''))) % 2

        test_str = test_str.rstrip('",')
        test_str += '"' * open_quotes + ']' * open_brackets + '}' * open_braces

        try:
            return json.loads(test_str)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not parse JSON: {json_str[:200]}")


class PhotoAnalysis(BaseModel):
    """General photo analysis results."""

    title: str = Field(description="Brief title describing the main subject")
    description: str = Field(description="Detailed description of the photo content")
    objects: list[str] = Field(description="List of main objects detected in the photo")
    colors: list[str] = Field(description="List of dominant colors")
    setting: str = Field(description="Type of setting or environment (e.g., indoor, outdoor, nature)")
    mood: str = Field(description="Overall mood or atmosphere of the photo")
    composition_notes: str = Field(description="Notes about composition and framing")


class ObjectDetection(BaseModel):
    """Detailed object detection results."""

    objects: list[dict] = Field(
        description="List of objects with name, count, and location description"
    )
    total_object_count: int = Field(description="Total number of distinct objects found")
    most_prominent: str = Field(description="The most prominent/central object")


class SceneAnalysis(BaseModel):
    """Scene and environment analysis results."""

    scene_type: str = Field(description="Type of scene (e.g., landscape, portrait, still life)")
    location_type: str = Field(description="Estimated location type (indoor/outdoor, urban/rural)")
    time_of_day: str = Field(description="Estimated time of day based on lighting")
    weather_conditions: str = Field(description="Visible weather conditions if any")
    depth_description: str = Field(description="Description of foreground, middle ground, and background")


def analyze_photo_general(image_path: str) -> PhotoAnalysis:
    """Perform general analysis of photo content.

    Args:
        image_path: Path to photo file (supports JPEG, PNG, HEIC, WebP, etc.)

    Returns:
        PhotoAnalysis object with comprehensive photo details

    Raises:
        FileNotFoundError: If image file does not exist
        ValueError: If format is not supported
    """
    image_path = Path(image_path)
    pil_img = open_image(image_path)

    m = start_session(
        model_id="hf.co/ibm-granite/granite-vision-4.1-4b-GGUF:Q4_K_M",
        model_options={
            ModelOption.CONTEXT_WINDOW: 4096,
            ModelOption.MAX_NEW_TOKENS: 1024,
        },
    )

    img = ImageBlock.from_pil_image(pil_img)

    prompt = """Analyze this photo. Return valid JSON with these fields:
{
    "title": "brief title",
    "description": "detailed description",
    "objects": ["obj1", "obj2"],
    "colors": ["color1", "color2"],
    "setting": "type of setting",
    "mood": "mood or atmosphere",
    "composition_notes": "composition notes"
}"""

    result = str(m.instruct(prompt, images=[img]))

    # Extract JSON from response
    json_match = re.search(r'\{.*', result, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON found in response: {result[:200]}")

    json_str = json_match.group(0)
    parsed_dict = fix_and_parse_json(json_str)

    # Ensure all required fields exist with defaults
    parsed_dict.setdefault('title', 'Photo Analysis')
    parsed_dict.setdefault('description', 'No description available')
    parsed_dict.setdefault('objects', [])
    parsed_dict.setdefault('colors', [])
    parsed_dict.setdefault('setting', 'Unknown')
    parsed_dict.setdefault('mood', 'Unknown')
    parsed_dict.setdefault('composition_notes', 'No notes')

    # Clean up list fields
    for key in ['objects', 'colors']:
        if isinstance(parsed_dict[key], list):
            parsed_dict[key] = [str(item).strip('"\' ') for item in parsed_dict[key]]

    parsed = PhotoAnalysis(**parsed_dict)
    return parsed


def analyze_objects(image_path: str) -> ObjectDetection:
    """Perform detailed object detection and analysis.

    Args:
        image_path: Path to photo file (supports JPEG, PNG, HEIC, WebP, etc.)

    Returns:
        ObjectDetection object with detailed object information

    Raises:
        FileNotFoundError: If image file does not exist
        ValueError: If format is not supported
    """
    image_path = Path(image_path)
    pil_img = open_image(image_path)

    m = start_session(
        model_id="hf.co/ibm-granite/granite-vision-4.1-4b-GGUF:Q4_K_M",
        model_options={
            ModelOption.CONTEXT_WINDOW: 4096,
            ModelOption.MAX_NEW_TOKENS: 1024,
        },
    )

    img = ImageBlock.from_pil_image(pil_img)

    prompt = """Return valid JSON for object detection:
{
    "objects": [{"name": "obj", "count": 1, "location": "center"}],
    "total_object_count": 5,
    "most_prominent": "main object"
}"""

    result = str(m.instruct(prompt, images=[img]))

    json_match = re.search(r'\{.*', result, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON found in response: {result[:200]}")

    json_str = json_match.group(0)
    parsed_dict = fix_and_parse_json(json_str)

    parsed_dict.setdefault('objects', [])
    parsed_dict.setdefault('total_object_count', len(parsed_dict.get('objects', [])))
    parsed_dict.setdefault('most_prominent', 'Unknown')

    parsed = ObjectDetection(**parsed_dict)
    return parsed


def analyze_scene(image_path: str) -> SceneAnalysis:
    """Perform scene and environment analysis.

    Args:
        image_path: Path to photo file (supports JPEG, PNG, HEIC, WebP, etc.)

    Returns:
        SceneAnalysis object with scene and environment details

    Raises:
        FileNotFoundError: If image file does not exist
        ValueError: If format is not supported
    """
    image_path = Path(image_path)
    pil_img = open_image(image_path)

    m = start_session(
        model_id="hf.co/ibm-granite/granite-vision-4.1-4b-GGUF:Q4_K_M",
        model_options={
            ModelOption.CONTEXT_WINDOW: 4096,
            ModelOption.MAX_NEW_TOKENS: 1024,
        },
    )

    img = ImageBlock.from_pil_image(pil_img)

    prompt = """Return valid JSON for scene analysis:
{
    "scene_type": "landscape/portrait/still life/etc",
    "location_type": "indoor/outdoor/urban/rural/nature",
    "time_of_day": "estimated time",
    "weather_conditions": "weather description",
    "depth_description": "foreground, middle ground, background"
}"""

    result = str(m.instruct(prompt, images=[img]))

    json_match = re.search(r'\{.*', result, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON found in response: {result[:200]}")

    json_str = json_match.group(0)
    parsed_dict = fix_and_parse_json(json_str)

    parsed_dict.setdefault('scene_type', 'Unknown')
    parsed_dict.setdefault('location_type', 'Unknown')
    parsed_dict.setdefault('time_of_day', 'Unknown')
    parsed_dict.setdefault('weather_conditions', 'Not visible')
    parsed_dict.setdefault('depth_description', 'No depth analysis')

    parsed = SceneAnalysis(**parsed_dict)
    return parsed


def print_general_analysis(analysis: PhotoAnalysis) -> None:
    """Pretty-print general photo analysis."""
    print("\n" + "=" * 60)
    print("PHOTO ANALYSIS")
    print("=" * 60)
    print(f"\nTitle: {analysis.title}")
    print(f"\nDescription:\n{analysis.description}")
    print(f"\nObjects: {', '.join(analysis.objects)}")
    print(f"Colors: {', '.join(analysis.colors)}")
    print(f"Setting: {analysis.setting}")
    print(f"Mood: {analysis.mood}")
    print(f"\nComposition:\n{analysis.composition_notes}")
    print("=" * 60 + "\n")


def print_object_detection(analysis: ObjectDetection) -> None:
    """Pretty-print object detection results."""
    print("\n" + "=" * 60)
    print("OBJECT DETECTION")
    print("=" * 60)
    print(f"\nMost Prominent: {analysis.most_prominent}")
    print(f"Total Object Types: {analysis.total_object_count}\n")

    print("Objects Detected:")
    for obj in analysis.objects:
        location = obj.get("location", "unknown location")
        count = obj.get("count", 1)
        name = obj.get("name", "unknown")
        print(f"  • {name} (x{count}) - {location}")

    print("=" * 60 + "\n")


def print_scene_analysis(analysis: SceneAnalysis) -> None:
    """Pretty-print scene analysis results."""
    print("\n" + "=" * 60)
    print("SCENE ANALYSIS")
    print("=" * 60)
    print(f"\nScene Type: {analysis.scene_type}")
    print(f"Location: {analysis.location_type}")
    print(f"Time of Day: {analysis.time_of_day}")
    print(f"Weather: {analysis.weather_conditions}")
    print(f"\nDepth Analysis:\n{analysis.depth_description}")
    print("=" * 60 + "\n")


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: uv run python photo_analyzer.py <image_path> [analysis_type]")
        print("\nSupported formats:")
        print("  JPEG, PNG, WebP" + (", HEIC/HEIF" if HEIC_SUPPORT else " (HEIC not available)"))
        print("\nAnalysis types:")
        print("  general  - comprehensive photo analysis (default)")
        print("  objects  - detailed object detection")
        print("  scene    - scene and environment analysis")
        if not HEIC_SUPPORT:
            print("\nTo enable HEIC support: pip install pillow-heif")
        sys.exit(1)

    image_path = sys.argv[1]
    analysis_type = sys.argv[2].lower() if len(sys.argv) > 2 else "general"

    try:
        if analysis_type == "objects":
            analysis = analyze_objects(image_path)
            print_object_detection(analysis)

        elif analysis_type == "scene":
            analysis = analyze_scene(image_path)
            print_scene_analysis(analysis)

        else:  # default to general
            analysis = analyze_photo_general(image_path)
            print_general_analysis(analysis)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        if "HEIC" in str(e):
            print("Install pillow-heif: pip install pillow-heif", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Analysis error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
