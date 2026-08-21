#!/usr/bin/env python3
"""
Advanced Photo Pipeline: Narrative-based Photo Matching
Combines photo selection, download, analysis, and intelligent narrative-based matching.

Uses Granite Vision 4.1 4B for photo analysis and Granite Vision 4.1 3B (via Ollama) for narrative matching.

Requirements:
    - Ollama running locally (default: http://localhost:11434)
    - Granite 4.1 3B model pulled: ollama pull granite4.1:3b

Usage:
    uv run python photo_pipeline_advanced.py [--fetch] [--analyze] [--view] [--narrative <text>] [--ollama-url <url>]

Examples:
    uv run python photo_pipeline_advanced.py --fetch --analyze --narrative "find photos of my dog playing outdoors"
    uv run python photo_pipeline_advanced.py --narrative "sunset with mountains" --view
    uv run python photo_pipeline_advanced.py --fetch --analyze --narrative "indoor scenes with people"
    uv run python photo_pipeline_advanced.py --narrative "outdoor" --view --ollama-url http://localhost:11434
"""

import os
import sys
import json
import time
import uuid
import re
import argparse
import webbrowser
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from typing import Optional, List, Dict, Any

from PIL import Image, ImageTk
from pydantic import BaseModel, Field

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import requests

from mellea import start_session
from mellea.core import ImageBlock
from mellea.backends.model_options import ModelOption
from mellea.backends.ollama import OllamaModelBackend as OllamaBackend
from mellea.stdlib.components import CBlock
from mellea.stdlib.context import SimpleContext
import mellea.stdlib.functional as mfuncs


# Constants
SCOPES = ['https://www.googleapis.com/auth/photospicker.mediaitems.readonly']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json'
PICKER_API_BASE_URL = 'https://photospicker.googleapis.com/v1'
DOWNLOADS_DIR = 'downloaded_photos'
METADATA_FILE = 'photo_metadata.json'


class PhotoAnalysis(BaseModel):
    """General photo analysis results."""
    title: str = Field(description="Brief title describing the main subject")
    description: str = Field(description="Detailed description of the photo content")
    objects: list[str] = Field(description="List of main objects detected in the photo")
    colors: list[str] = Field(description="List of dominant colors")
    setting: str = Field(description="Type of setting or environment")
    mood: str = Field(description="Overall mood or atmosphere of the photo")
    composition_notes: str = Field(description="Notes about composition and framing")


class NarrativeMatch(BaseModel):
    """Narrative matching result."""
    matches: bool = Field(description="Whether photo matches the narrative")
    confidence: float = Field(description="Confidence score 0.0-1.0")
    reasoning: str = Field(description="Explanation of why it matches or doesn't match")
    key_elements: list[str] = Field(description="Which elements in the photo match the narrative")


class PhotoMetadata(BaseModel):
    """Metadata for downloaded photo with analysis."""
    filename: str
    local_path: str
    google_id: str
    created_time: str
    analysis: Optional[PhotoAnalysis] = None
    narrative_match: Optional[NarrativeMatch] = None


class GooglePhotosPicker:
    """Client for Google Photos Picker API."""

    def __init__(self, credentials_file: str = CREDENTIALS_FILE):
        self.credentials_file = credentials_file
        self.credentials = None
        self.access_token = None

    def authenticate(self) -> str:
        try:
            if os.path.exists(TOKEN_FILE):
                self.credentials = Credentials.from_authorized_user_file(
                    TOKEN_FILE, SCOPES
                )
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES
                )
                self.credentials = flow.run_local_server(port=0)
                with open(TOKEN_FILE, 'w') as token:
                    token.write(self.credentials.to_json())

        except Exception as e:
            raise Exception(f"Authentication failed: {e}")

        if self.credentials.expired and self.credentials.refresh_token:
            self.credentials.refresh(Request())

        self.access_token = self.credentials.token
        return self.access_token

    def get_headers(self) -> Dict[str, str]:
        if not self.access_token:
            raise Exception("Not authenticated. Call authenticate() first.")
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }

    def create_picker_session(self) -> Dict[str, Any]:
        headers = self.get_headers()
        request_id = str(uuid.uuid4())
        params = {'requestId': request_id}

        try:
            response = requests.post(
                f'{PICKER_API_BASE_URL}/sessions',
                headers=headers,
                json={},
                params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to create picker session: {e}")

    def open_picker_in_browser(self, picker_uri: str) -> None:
        print(f"\n🌐 Opening Google Photos picker in your browser...")
        print(f"📸 Please select the photos you want to access")
        print(f"⚠️  After selecting photos, return to this terminal")
        webbrowser.open(picker_uri)

    def poll_session_status(
        self,
        session_id: str,
        timeout: int = 600,
        poll_interval: float = 2
    ) -> Optional[Dict[str, Any]]:
        headers = self.get_headers()
        start_time = time.time()
        elapsed = 0

        while elapsed < timeout:
            try:
                response = requests.get(
                    f'{PICKER_API_BASE_URL}/sessions/{session_id}',
                    headers=headers
                )
                response.raise_for_status()
                session_data = response.json()

                if session_data.get('mediaItemsSet', False):
                    print(f"\n✓ Photos selected!")
                    return session_data

                recommended_interval = session_data.get('recommendedPollingIntervalMs', poll_interval * 1000)
                poll_interval = recommended_interval / 1000

                elapsed = time.time() - start_time
                remaining = timeout - elapsed
                print(f"⏳ Waiting for selection... ({int(remaining)}s remaining)")
                time.sleep(poll_interval)
                elapsed = time.time() - start_time

            except requests.exceptions.RequestException as e:
                print(f"⚠️  Polling error: {e}")
                time.sleep(poll_interval)
                elapsed = time.time() - start_time

        print("❌ Polling timeout reached")
        return None

    def get_selected_media(self, session_id: str, page_size: int = 100) -> List[Dict[str, Any]]:
        headers = self.get_headers()
        params = {
            'sessionId': session_id,
            'pageSize': min(page_size, 100)
        }

        try:
            response = requests.get(
                f'{PICKER_API_BASE_URL}/mediaItems',
                headers=headers,
                params=params
            )
            response.raise_for_status()
            return response.json().get('mediaItems', [])
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get selected media: {e}")

    def download_photo(self, base_url: str, filename: str, size: str = 'w1920-h1080') -> bool:
        try:
            download_url = f"{base_url}={size}" if size else base_url
            headers = self.get_headers()
            response = requests.get(download_url, headers=headers)

            if response.status_code == 403:
                response = requests.get(download_url)

            response.raise_for_status()

            os.makedirs(DOWNLOADS_DIR, exist_ok=True)
            filepath = os.path.join(DOWNLOADS_DIR, filename)

            with open(filepath, 'wb') as f:
                f.write(response.content)

            print(f"✓ Downloaded: {filename}")
            return True
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            return False


def fix_and_parse_json(json_str: str) -> dict:
    """Attempt to fix and parse malformed JSON from model response."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    json_str = re.sub(r'\["+"', '["', json_str)
    json_str = re.sub(r'""\]', '"]', json_str)
    json_str = re.sub(r'""+,', '",', json_str)
    json_str = re.sub(r', ""+', ', "', json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    for end_pos in range(len(json_str) - 1, max(len(json_str) - 500, 0), -1):
        test_str = json_str[:end_pos]

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


def open_image(image_path: Path) -> Image.Image:
    """Open image file with HEIC/HEIF support."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    suffix = image_path.suffix.lower()

    if suffix in ['.heic', '.heif']:
        if not HEIC_SUPPORT:
            raise ValueError(
                f"HEIC format not supported. Install pillow-heif: pip install pillow-heif"
            )

    try:
        img = Image.open(image_path)
        img.load()
        return img
    except Exception as e:
        raise ValueError(f"Failed to open image {image_path}: {e}")


def analyze_photo(image_path: str) -> PhotoAnalysis:
    """Perform general analysis of photo content using Granite Vision 4.1 4B."""
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

    json_match = re.search(r'\{.*', result, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON found in response: {result[:200]}")

    json_str = json_match.group(0)
    parsed_dict = fix_and_parse_json(json_str)

    parsed_dict.setdefault('title', 'Photo Analysis')
    parsed_dict.setdefault('description', 'No description available')
    parsed_dict.setdefault('objects', [])
    parsed_dict.setdefault('colors', [])
    parsed_dict.setdefault('setting', 'Unknown')
    parsed_dict.setdefault('mood', 'Unknown')
    parsed_dict.setdefault('composition_notes', 'No notes')

    for key in ['objects', 'colors']:
        if isinstance(parsed_dict[key], list):
            parsed_dict[key] = [str(item).strip('"\' ') for item in parsed_dict[key]]

    parsed = PhotoAnalysis(**parsed_dict)
    return parsed


def match_narrative(analysis: PhotoAnalysis, narrative: str, ollama_url: str = "http://localhost:11434") -> NarrativeMatch:
    """Match photo analysis against narrative condition using Granite Vision 4.1 3B model via Ollama."""
    backend = OllamaBackend(
        model_id="granite4.1:3b",
        base_url=ollama_url,
    )

    ctx = SimpleContext()

    model_options = {
        ModelOption.TEMPERATURE: 0.3,
        ModelOption.MAX_NEW_TOKENS: 512,
    }

    photo_info = f"""
Title: {analysis.title}
Description: {analysis.description}
Objects: {', '.join(analysis.objects)}
Colors: {', '.join(analysis.colors)}
Setting: {analysis.setting}
Mood: {analysis.mood}
Composition: {analysis.composition_notes}
"""

    prompt = f"""Given this photo analysis:
{photo_info}

Does this photo match this narrative condition: "{narrative}"?

Return valid JSON:
{{
    "matches": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "explanation",
    "key_elements": ["element1", "element2"]
}}

Be strict and accurate. Only mark as matches=true if the photo genuinely fits the narrative."""

    try:
        action = CBlock(prompt)
        mot, gen_ctx = mfuncs.act(
            action, ctx, backend, strategy=None, model_options=model_options
        )

        result = str(mot.value)

        json_match = re.search(r'\{.*', result, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {result[:200]}")

        json_str = json_match.group(0)
        parsed_dict = fix_and_parse_json(json_str)

        parsed_dict.setdefault('matches', False)
        parsed_dict.setdefault('confidence', 0.0)
        parsed_dict.setdefault('reasoning', 'Unable to determine')
        parsed_dict.setdefault('key_elements', [])

        # Ensure confidence is a float
        try:
            parsed_dict['confidence'] = float(parsed_dict['confidence'])
        except (ValueError, TypeError):
            parsed_dict['confidence'] = 0.0

        parsed = NarrativeMatch(**parsed_dict)
        return parsed

    except Exception as e:
        raise ValueError(f"Failed to match narrative: {e}")


def download_and_analyze(picker: GooglePhotosPicker, media_items: List[Dict], analyze: bool = False):
    """Download photos and optionally analyze them, save metadata."""
    metadata_list = []
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)

    for idx, item in enumerate(media_items, 1):
        media_file = item.get('mediaFile', {})
        item_type = item.get('type', '')

        if item_type != 'PHOTO' or not media_file.get('baseUrl'):
            print(f"⏭️  Skipping item {idx} (type: {item_type})")
            continue

        base_url = media_file.get('baseUrl')
        filename = media_file.get('filename', f'photo_{idx}.jpg')

        if picker.download_photo(base_url, filename):
            local_path = os.path.join(DOWNLOADS_DIR, filename)

            photo_meta = PhotoMetadata(
                filename=filename,
                local_path=local_path,
                google_id=item.get('id', ''),
                created_time=item.get('createTime', '')
            )

            if analyze:
                try:
                    print(f"🔍 Analyzing {filename}...")
                    analysis = analyze_photo(local_path)
                    photo_meta.analysis = analysis
                    print(f"✓ Analysis complete for {filename}")
                except Exception as e:
                    print(f"⚠️  Analysis failed for {filename}: {e}")

            metadata_list.append(photo_meta.model_dump())

    # Save metadata
    if metadata_list:
        metadata_path = os.path.join(DOWNLOADS_DIR, METADATA_FILE)
        with open(metadata_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        print(f"\n✓ Saved metadata to {metadata_path}")

    return metadata_list


def load_metadata() -> List[Dict]:
    """Load photo metadata from file."""
    metadata_path = os.path.join(DOWNLOADS_DIR, METADATA_FILE)
    if not os.path.exists(metadata_path):
        print("No metadata file found. Use --fetch to get photos from Google Photos")
        return []

    with open(metadata_path, 'r') as f:
        return json.load(f)


def match_photos_to_narrative(metadata_list: List[Dict], narrative: str, ollama_url: str = "http://localhost:11434") -> List[Dict]:
    """Match all photos against narrative condition."""
    matched_photos = []

    for photo in metadata_list:
        if not photo.get('analysis'):
            print(f"⏭️  {photo['filename']} not analyzed, skipping")
            continue

        try:
            analysis_dict = photo['analysis']
            analysis = PhotoAnalysis(**analysis_dict)

            print(f"🔗 Matching '{photo['filename']}' against narrative...")
            match = match_narrative(analysis, narrative, ollama_url)
            photo['narrative_match'] = match.model_dump()

            if match.matches:
                print(f"✓ MATCH! Confidence: {match.confidence:.1%}")
                matched_photos.append(photo)
            else:
                print(f"✗ No match (confidence: {match.confidence:.1%})")

        except Exception as e:
            print(f"⚠️  Matching failed for {photo['filename']}: {e}")

    # Save updated metadata
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)
    metadata_path = os.path.join(DOWNLOADS_DIR, METADATA_FILE)
    with open(metadata_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)

    return matched_photos


class AdvancedPhotoViewer:
    """GUI for viewing photos with narrative matching results."""

    def __init__(self, root, photos: List[Dict], narrative: Optional[str] = None):
        self.root = root
        self.root.title("Advanced Photo Viewer - Narrative Matching")
        self.root.geometry("1100x900")
        self.root.minsize(900, 700)

        self.photos = photos
        self.current_index = 0
        self.photo_image = None
        self.is_fullscreen = False
        self.narrative = narrative

        if not self.photos:
            messagebox.showerror("Error", "No photos to display")
            self.root.quit()
            return

        self.setup_ui()
        self.display_photo()

    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#e0e0e0", padx=10, pady=10)
        header_frame.pack(fill=tk.X)

        title = "Advanced Photo Viewer"
        if self.narrative:
            title += f" - Narrative: '{self.narrative}'"
        title += f" [{len(self.photos)} photos]"

        tk.Label(header_frame, text=title, font=("Arial", 12, "bold"), bg="#e0e0e0").pack(side=tk.LEFT)

        # Image display
        self.image_label = tk.Label(self.root, bg="black", text="Loading...")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Navigation
        nav_frame = tk.Frame(self.root, bg="#f0f0f0", padx=10, pady=10)
        nav_frame.pack(fill=tk.X)

        tk.Button(nav_frame, text="Previous", command=self.show_previous).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Next", command=self.show_next).pack(side=tk.LEFT, padx=5)

        self.info_label = tk.Label(nav_frame, text="", bg="#f0f0f0")
        self.info_label.pack(side=tk.LEFT, padx=20, fill=tk.X, expand=True)

        tk.Button(nav_frame, text="Analysis", command=self.show_analysis).pack(side=tk.RIGHT, padx=5)
        tk.Button(nav_frame, text="Narrative Match", command=self.show_narrative_match).pack(side=tk.RIGHT, padx=5)
        tk.Button(nav_frame, text="Exit", command=self.root.quit).pack(side=tk.RIGHT, padx=5)

        self.root.bind('<Right>', lambda e: self.show_next())
        self.root.bind('<Left>', lambda e: self.show_previous())
        self.root.bind('<a>', lambda e: self.show_analysis())
        self.root.bind('<n>', lambda e: self.show_narrative_match())
        self.root.bind('<q>', lambda e: self.root.quit())

    def display_photo(self):
        if not self.photos:
            self.image_label.config(text="No photos loaded", image="")
            return

        try:
            photo_data = self.photos[self.current_index]
            photo_path = photo_data.get('local_path')

            if not os.path.exists(photo_path):
                self.image_label.config(text=f"File not found: {photo_path}", bg="black")
                return

            image = Image.open(photo_path)

            window_width = self.image_label.winfo_width()
            window_height = self.image_label.winfo_height()

            if window_width <= 1:
                window_width = 900
                window_height = 600

            image.thumbnail((window_width, window_height), Image.Resampling.LANCZOS)

            self.photo_image = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.photo_image, text="")

            filename = Path(photo_path).name
            self.info_label.config(
                text=f"[{self.current_index + 1}/{len(self.photos)}] {filename}"
            )
        except Exception as e:
            self.image_label.config(text=f"Error loading image: {e}", bg="black")

    def show_analysis(self):
        photo_data = self.photos[self.current_index]
        analysis = photo_data.get('analysis')

        if not analysis:
            messagebox.showinfo("Analysis", "No analysis available for this photo")
            return

        details = f"""
PHOTO ANALYSIS

Title: {analysis.get('title', 'N/A')}

Description:
{analysis.get('description', 'N/A')}

Objects: {', '.join(analysis.get('objects', []))}
Colors: {', '.join(analysis.get('colors', []))}

Setting: {analysis.get('setting', 'N/A')}
Mood: {analysis.get('mood', 'N/A')}

Composition:
{analysis.get('composition_notes', 'N/A')}
        """

        messagebox.showinfo("Photo Analysis", details)

    def show_narrative_match(self):
        photo_data = self.photos[self.current_index]
        match = photo_data.get('narrative_match')

        if not match:
            messagebox.showinfo("Narrative Match", "No narrative matching data available")
            return

        status = "✓ MATCHES" if match.get('matches') else "✗ NO MATCH"
        details = f"""{status}

Narrative: "{self.narrative}"

Confidence: {match.get('confidence', 0):.1%}

Reasoning:
{match.get('reasoning', 'N/A')}

Key Elements:
{', '.join(match.get('key_elements', []))}
        """

        messagebox.showinfo("Narrative Matching Result", details)

    def show_next(self):
        if self.photos:
            self.current_index = (self.current_index + 1) % len(self.photos)
            self.display_photo()

    def show_previous(self):
        if self.photos:
            self.current_index = (self.current_index - 1) % len(self.photos)
            self.display_photo()


def main():
    parser = argparse.ArgumentParser(description="Advanced Photo Pipeline: Narrative-based Photo Matching")
    parser.add_argument('--fetch', action='store_true', help='Fetch photos from Google Photos (select and download)')
    parser.add_argument('--analyze', action='store_true', help='Analyze photos (existing or fetched)')
    parser.add_argument('--narrative', type=str, help='Narrative condition to match (e.g., "dog playing outdoors")')
    parser.add_argument('--view', action='store_true', help='View photos with GUI')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434', help='Ollama server URL for narrative matching')

    args = parser.parse_args()

    print("="*70)
    print("ADVANCED PHOTO PIPELINE - Narrative-based Photo Matching")
    print("="*70)

    metadata_list = []

    # Step 1: Optional fetch from Google Photos
    if args.fetch:
        picker = GooglePhotosPicker()

        print("\nStep 1: Authenticating with Google...")
        try:
            picker.authenticate()
            print("✓ Authentication successful")
        except Exception as e:
            print(f"✗ Authentication failed: {e}")
            return

        print("\nStep 2: Creating picker session...")
        try:
            session = picker.create_picker_session()
            session_id = session.get('id')
            picker_uri = session.get('pickerUri')
            print(f"✓ Session created")
        except Exception as e:
            print(f"✗ Failed to create session: {e}")
            return

        print("\nStep 3: Opening Google Photos picker...")
        picker.open_picker_in_browser(picker_uri)

        print("\nStep 4: Waiting for photo selection...")
        session_result = picker.poll_session_status(session_id)

        if not session_result:
            print("✗ No photos selected within timeout")
            return

        print("\nStep 5: Retrieving selected media details...")
        try:
            media_items = picker.get_selected_media(session_id)
            print(f"✓ Retrieved {len(media_items)} item(s)")
        except Exception as e:
            print(f"✗ Failed to retrieve media: {e}")
            return

        print("\nStep 6: Downloading photos...")
        metadata_list = download_and_analyze(picker, media_items, analyze=False)

        if not metadata_list:
            print("✗ No photos downloaded")
            return

    # Step 2: Work with existing photos if no fetch
    if not metadata_list and (args.analyze or args.narrative or args.view):
        print("\nLoading existing photos from metadata...")
        metadata_list = load_metadata()
        if not metadata_list:
            print("✗ No photos found. Use --fetch to get photos from Google Photos")
            return
        print(f"✓ Loaded {len(metadata_list)} photos")

    # Step 3: Analyze photos if requested
    if args.analyze and metadata_list:
        print("\nAnalyzing photos...")
        for idx, photo in enumerate(metadata_list, 1):
            if photo.get('analysis'):
                print(f"⏭️  {photo['filename']} already analyzed, skipping")
                continue

            local_path = photo.get('local_path')
            if not os.path.exists(local_path):
                print(f"⚠️  File not found: {local_path}, skipping")
                continue

            try:
                print(f"🔍 Analyzing {photo['filename']}... ({idx}/{len(metadata_list)})")
                analysis = analyze_photo(local_path)
                photo['analysis'] = analysis.model_dump()
                print(f"✓ Analysis complete")
            except Exception as e:
                print(f"⚠️  Analysis failed: {e}")

        # Save updated metadata
        os.makedirs(DOWNLOADS_DIR, exist_ok=True)
        metadata_path = os.path.join(DOWNLOADS_DIR, METADATA_FILE)
        with open(metadata_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        print(f"✓ Saved metadata to {metadata_path}")

    # Step 4: Match photos to narrative
    matched_photos = []
    if args.narrative:
        if not all(photo.get('analysis') for photo in metadata_list):
            print("\n⚠️  Some photos lack analysis. Running analysis first...")
            for photo in metadata_list:
                if not photo.get('analysis'):
                    local_path = photo.get('local_path')
                    if os.path.exists(local_path):
                        try:
                            print(f"🔍 Analyzing {photo['filename']}...")
                            analysis = analyze_photo(local_path)
                            photo['analysis'] = analysis.model_dump()
                        except Exception as e:
                            print(f"⚠️  Analysis failed: {e}")

        print(f"\n🔗 Matching photos to narrative: '{args.narrative}'")
        print(f"Using Ollama at: {args.ollama_url}")
        matched_photos = match_photos_to_narrative(metadata_list, args.narrative, args.ollama_url)
        print(f"\n✓ Matched {len(matched_photos)}/{len(metadata_list)} photos")

        if matched_photos:
            metadata_list = matched_photos

    # Step 5: View with optional narrative filtering
    if (args.view or args.narrative) and metadata_list:
        print("\nPreparing viewer...")
        root = tk.Tk()
        app = AdvancedPhotoViewer(root, metadata_list, args.narrative)
        root.mainloop()
    elif not args.view and not args.narrative and metadata_list:
        print(f"\n✓ Operation complete! {len(metadata_list)} photos available")
        print("\nUsage tips:")
        print("  --view                                  View photos in GUI")
        print("  --narrative '<condition>'               Match and view by narrative")
        print("  --analyze                               Analyze photos")
        print("  --fetch --analyze --narrative '<text>'  Full workflow")


if __name__ == '__main__':
    main()
