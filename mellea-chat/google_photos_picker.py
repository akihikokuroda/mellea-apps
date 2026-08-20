#!/usr/bin/env python3
"""
Google Photos Picker API Sample
Demonstrates how to use the Google Photos Picker API to let users select photos,
then access them programmatically.
"""

import os
import json
import time
import uuid
import webbrowser
from typing import Optional, List, Dict, Any
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import requests


# Configuration
SCOPES = ['https://www.googleapis.com/auth/photospicker.mediaitems.readonly']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json'
PICKER_API_BASE_URL = 'https://photospicker.googleapis.com/v1'


class GooglePhotosPicker:
    """Client for Google Photos Picker API."""

    def __init__(self, credentials_file: str = CREDENTIALS_FILE):
        """Initialize the Google Photos Picker client."""
        self.credentials_file = credentials_file
        self.credentials = None
        self.access_token = None

    def authenticate(self) -> str:
        """
        Authenticate with Google using OAuth 2.0.
        Returns the access token.
        """
        try:
            # Try to load existing token
            if os.path.exists(TOKEN_FILE):
                self.credentials = Credentials.from_authorized_user_file(
                    TOKEN_FILE, SCOPES
                )
            else:
                # Create new OAuth flow
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES
                )
                self.credentials = flow.run_local_server(port=0)

                # Save token for future use
                with open(TOKEN_FILE, 'w') as token:
                    token.write(self.credentials.to_json())

        except Exception as e:
            raise Exception(f"Authentication failed: {e}")

        # Refresh if expired
        if self.credentials.expired and self.credentials.refresh_token:
            self.credentials.refresh(Request())

        self.access_token = self.credentials.token
        return self.access_token

    def get_headers(self) -> Dict[str, str]:
        """Get authorization headers for API requests."""
        if not self.access_token:
            raise Exception("Not authenticated. Call authenticate() first.")
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }

    def create_picker_session(self) -> Dict[str, Any]:
        """
        Create a new picker session.
        Returns session details including the picker URL.
        """
        headers = self.get_headers()
        payload = {}

        # Add requestId as query parameter
        request_id = str(uuid.uuid4())
        params = {'requestId': request_id}

        try:
            response = requests.post(
                f'{PICKER_API_BASE_URL}/sessions',
                headers=headers,
                json=payload,
                params=params
            )
            response.raise_for_status()
            session_data = response.json()
            return session_data
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to create picker session: {e}")

    def open_picker_in_browser(self, picker_uri: str) -> None:
        """
        Open the picker URI in the default browser.

        Args:
            picker_uri: The URI from the session
        """
        print(f"\n🌐 Opening Google Photos picker in your browser...")
        print(f"📸 Please select the photos you want to access")
        print(f"\n⚠️  IMPORTANT: After selecting photos, return to this terminal")
        webbrowser.open(picker_uri)

    def poll_session_status(
        self,
        session_id: str,
        timeout: int = 600,
        poll_interval: float = 2
    ) -> Optional[Dict[str, Any]]:
        """
        Poll the session status until media items are selected or timeout occurs.

        Args:
            session_id: The picker session ID
            timeout: Maximum time to poll in seconds (default 10 minutes)
            poll_interval: Time between polls in seconds (default 2)

        Returns:
            Session data with mediaItemsSet=true, or None if timeout
        """
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

                # Check if media items have been selected
                if session_data.get('mediaItemsSet', False):
                    print(f"\n✓ Photos selected!")
                    return session_data

                # Get recommended poll interval from response if available
                recommended_interval = session_data.get('recommendedPollingIntervalMs', poll_interval * 1000)
                poll_interval = recommended_interval / 1000

                elapsed = time.time() - start_time
                remaining = timeout - elapsed
                print(f"⏳ Waiting for selection... ({int(remaining)}s remaining, polling every {poll_interval:.1f}s)")
                time.sleep(poll_interval)
                elapsed = time.time() - start_time

            except requests.exceptions.RequestException as e:
                print(f"⚠️  Polling error: {e}")
                time.sleep(poll_interval)
                elapsed = time.time() - start_time

        print("❌ Polling timeout reached")
        return None

    def get_selected_media(self, session_id: str, page_size: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve details of selected media items.

        Args:
            session_id: The picker session ID
            page_size: Number of items to retrieve per page

        Returns:
            List of selected media items with metadata
        """
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
            media_data = response.json()
            return media_data.get('mediaItems', [])
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get selected media: {e}")

    def download_photo(self, base_url: str, filename: str, size: str = 'w800-h800') -> bool:
        """
        Download a photo from Google Photos.

        Args:
            base_url: Base URL from media item
            filename: Local filename to save to
            size: Size parameter (e.g., 'w800-h800', 'w1920-h1080')

        Returns:
            True if successful, False otherwise
        """
        try:
            # Append size parameters to the base URL
            download_url = f"{base_url}={size}" if size else base_url

            # Try with authorization header first
            headers = self.get_headers()
            response = requests.get(download_url, headers=headers)

            # If that fails with 403, try without headers (some URLs are public)
            if response.status_code == 403:
                response = requests.get(download_url)

            response.raise_for_status()

            with open(filename, 'wb') as f:
                f.write(response.content)

            print(f"✓ Downloaded: {filename}")
            return True
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            return False

    def display_media_info(self, media_items: List[Dict[str, Any]]) -> None:
        """Display information about media items."""
        print("\n" + "="*70)
        print("SELECTED MEDIA ITEMS")
        print("="*70)

        for idx, item in enumerate(media_items, 1):
            media_file = item.get('mediaFile', {})
            media_metadata = media_file.get('mediaFileMetadata', {})
            filename = media_file.get('filename', item.get('filename', 'Unknown'))

            print(f"\n[{idx}] {filename}")
            print(f"    ID: {item.get('id', 'N/A')[:50]}...")
            print(f"    Created: {item.get('createTime', 'N/A')}")
            print(f"    Type: {item.get('type', 'Unknown')}")

            if media_metadata:
                if media_metadata.get('width'):
                    print(f"    Resolution: {media_metadata.get('width')}x{media_metadata.get('height')}")
                if media_metadata.get('cameraMake'):
                    print(f"    Camera: {media_metadata.get('cameraMake')} {media_metadata.get('cameraModel', '')}")

            base_url = media_file.get('baseUrl', item.get('baseUrl', 'N/A'))
            print(f"    URL: {base_url[:60]}...")


def main():
    """Main function demonstrating the Google Photos Picker API workflow."""

    print("="*70)
    print("GOOGLE PHOTOS PICKER API - INTERACTIVE DEMO")
    print("="*70)

    # Initialize picker client
    picker = GooglePhotosPicker()

    # Step 1: Authenticate
    print("\nStep 1: Authenticating with Google...")
    try:
        token = picker.authenticate()
        print(f"✓ Authentication successful")
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        return

    # Step 2: Create picker session
    print("\nStep 2: Creating picker session...")
    try:
        session = picker.create_picker_session()
        session_id = session.get('id')  # Use 'id' instead of 'sessionId'
        picker_uri = session.get('pickerUri')
        print(f"✓ Session created: {session_id}")
    except Exception as e:
        print(f"✗ Failed to create session: {e}")
        print("\n⚠️  TROUBLESHOOTING:")
        print("  1. Make sure you enabled the 'Photos Library API' in Google Cloud Console")
        print("  2. Check that your OAuth consent screen has 'photospicker.mediaitems.readonly' scope")
        print("  3. Verify your email is added as a test user")
        print("  4. Try deleting token.json and re-authenticating")
        print("  5. Check Google Cloud Console for any API usage quota issues")
        return

    # Step 3: Open picker in browser
    print("\nStep 3: Opening Google Photos picker...")
    picker.open_picker_in_browser(picker_uri)

    # Step 4: Poll for selection
    print("\nStep 4: Waiting for photo selection...")
    session_result = picker.poll_session_status(session_id)

    if not session_result:
        print("✗ No photos selected within timeout")
        return

    # Step 5: Get selected media details
    print("\nStep 5: Retrieving selected media details...")
    try:
        media_items = picker.get_selected_media(session_id)
        print(f"✓ Retrieved {len(media_items)} item(s)")
        picker.display_media_info(media_items)
    except Exception as e:
        print(f"✗ Failed to retrieve media: {e}")
        return

    # Step 6: Optional - Download selected photos
    print("\n" + "="*70)
    print("Step 6: Downloading photos...")
    print("="*70)

    downloaded_count = 0
    for idx, item in enumerate(media_items, 1):
        media_file = item.get('mediaFile', {})
        item_type = item.get('type', '')

        if item_type == 'PHOTO' and media_file.get('baseUrl'):
            try:
                base_url = media_file.get('baseUrl')
                filename = f"photo_{idx}_{media_file.get('filename', 'photo')}"
                if picker.download_photo(base_url, filename):
                    downloaded_count += 1
            except Exception as e:
                print(f"✗ Failed to download item {idx}: {e}")
        else:
            print(f"⏭️  Skipping item {idx} (type: {item_type})")

    print("\n" + "="*70)
    print(f"✓ Complete! Downloaded {downloaded_count}/{len(media_items)} photos")
    print("="*70)


if __name__ == '__main__':
    main()
