#!/usr/bin/env python3
"""
Integrated Photo Pipeline: Select, Download, Analyze and View
Combines Google Photos Picker, Photo Analyzer, and Photo Viewer functionality.

Usage:
    uv run python photo_pipeline.py [--analyze] [--view] [--condition <condition>]

Examples:
    uv run python photo_pipeline.py --analyze
    uv run python photo_pipeline.py --analyze --view
    uv run python photo_pipeline.py --view --condition "outdoor"
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


class PhotoMetadata(BaseModel):
    """Metadata for downloaded photo with analysis."""
    filename: str
    local_path: str
    google_id: str
    created_time: str
    analysis: Optional[PhotoAnalysis] = None


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
    """Perform general analysis of photo content."""
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
        print("No metadata file found. Run with --analyze flag first.")
        return []

    with open(metadata_path, 'r') as f:
        return json.load(f)


def filter_photos_by_condition(metadata_list: List[Dict], condition: str) -> List[Dict]:
    """Filter photos by condition in metadata."""
    if not condition:
        return metadata_list

    filtered = []
    condition_lower = condition.lower()

    for photo in metadata_list:
        if not photo.get('analysis'):
            continue

        analysis = photo['analysis']

        # Search in various fields
        searchable_fields = [
            analysis.get('title', ''),
            analysis.get('description', ''),
            analysis.get('setting', ''),
            analysis.get('mood', ''),
            ' '.join(analysis.get('objects', [])),
            ' '.join(analysis.get('colors', []))
        ]

        if any(condition_lower in field.lower() for field in searchable_fields):
            filtered.append(photo)

    return filtered


class PhotoViewer:
    """GUI for viewing filtered photos."""

    def __init__(self, root, photos: List[Dict], condition: Optional[str] = None):
        self.root = root
        self.root.title("Photo Viewer - Pipeline")
        self.root.geometry("1000x800")
        self.root.minsize(800, 600)

        self.photos = photos
        self.current_index = 0
        self.photo_image = None
        self.is_fullscreen = False
        self.condition = condition

        if not self.photos:
            messagebox.showerror("Error", "No photos to display")
            self.root.quit()
            return

        self.setup_ui()
        self.display_photo()

    def setup_ui(self):
        # Header with condition info
        header_frame = tk.Frame(self.root, bg="#e0e0e0", padx=10, pady=10)
        header_frame.pack(fill=tk.X)

        title = f"Photo Viewer - Pipeline"
        if self.condition:
            title += f" (Filter: '{self.condition}')"
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

        tk.Button(nav_frame, text="Details", command=self.show_details).pack(side=tk.RIGHT, padx=5)
        tk.Button(nav_frame, text="Exit", command=self.root.quit).pack(side=tk.RIGHT, padx=5)

        self.root.bind('<Right>', lambda e: self.show_next())
        self.root.bind('<Left>', lambda e: self.show_previous())
        self.root.bind('<d>', lambda e: self.show_details())
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
                window_width = 800
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

    def show_details(self):
        photo_data = self.photos[self.current_index]
        analysis = photo_data.get('analysis')

        if not analysis:
            messagebox.showinfo("Details", "No analysis available for this photo")
            return

        details = f"""
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

        messagebox.showinfo("Photo Details", details)

    def show_next(self):
        if self.photos:
            self.current_index = (self.current_index + 1) % len(self.photos)
            self.display_photo()

    def show_previous(self):
        if self.photos:
            self.current_index = (self.current_index - 1) % len(self.photos)
            self.display_photo()


def main():
    parser = argparse.ArgumentParser(description="Photo Pipeline: Select, Download, Analyze and View")
    parser.add_argument('--fetch', action='store_true', help='Fetch photos from Google Photos (select and download)')
    parser.add_argument('--analyze', action='store_true', help='Analyze photos (existing or fetched)')
    parser.add_argument('--view', action='store_true', help='View photos with GUI')
    parser.add_argument('--condition', type=str, help='Filter photos by condition (e.g., "outdoor", "dog")')

    args = parser.parse_args()

    print("="*70)
    print("PHOTO PIPELINE - Select, Download, Analyze and View")
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
    if not metadata_list and (args.analyze or args.view or args.condition):
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

    # Step 4: View with optional filtering
    if (args.view or args.condition) and metadata_list:
        print("\nPreparing viewer...")

        if args.condition:
            filtered = filter_photos_by_condition(metadata_list, args.condition)
            print(f"✓ Filtered to {len(filtered)}/{len(metadata_list)} photos matching '{args.condition}'")
            if not filtered:
                print("✗ No photos match the condition")
                return
            metadata_list = filtered

        root = tk.Tk()
        app = PhotoViewer(root, metadata_list, args.condition)
        root.mainloop()
    elif not args.view and not args.condition and metadata_list:
        print(f"\n✓ Operation complete! {len(metadata_list)} photos available")
        print("\nUsage tips:")
        print("  --view                    View photos in GUI")
        print("  --condition <keyword>     Filter by keyword and view")
        print("  --analyze                 Analyze photos")


if __name__ == '__main__':
    main()
