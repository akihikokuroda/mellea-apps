# Google Photos Picker API - Python Setup Guide

This guide walks you through setting up and using the Google Photos Picker API with Python to let users select photos from their Google Photos library.

## Prerequisites

- Python 3.7+
- A Google Cloud Project with the Photos Library API enabled
- OAuth 2.0 credentials (Desktop application type)
- A test user account added to your project (in OAuth consent screen)

## Why the Picker API?

As of April 1, 2025, Google deprecated direct access to users' Google Photos libraries for privacy reasons. The **Picker API is the new official way** to access user photos. It gives users control by making them explicitly select the photos they want to share with your app.

## Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the **Photos Library API**:
   - In the search bar, type "Photos Library API"
   - Click on it and press "Enable"

## Step 2: Set Up OAuth 2.0 Credentials

1. In the Google Cloud Console, go to **APIs & Services** > **Credentials**
2. Click **Create Credentials** > **OAuth client ID**
3. If prompted, configure the OAuth consent screen first:
   - User type: **External** (for testing)
   - Fill in the required fields
   - Under "Scopes", add `https://www.googleapis.com/auth/photospicker.mediaitems.readonly`
4. Back to OAuth client ID creation:
   - Choose **Desktop application** as the application type
5. Download the credentials as JSON
6. Save it as `credentials.json` in the same directory as the Python script

## Step 3: Add Test Users

1. In Google Cloud Console, go to **APIs & Services** > **OAuth consent screen**
2. Under "Test users", click **Add users**
3. Add your Google account email and any other test accounts
4. These accounts can now use the app during development

## Step 4: Install Dependencies

```bash
pip install -r requirements_photos.txt
```

Or install individually:

```bash
pip install google-auth google-auth-oauthlib requests
```

## Step 5: Run the Application

```bash
python3 google_photos_picker.py
```

### Workflow

1. **Authentication**: Script prompts you to authenticate with Google (first run only)
2. **Session Creation**: A picker session is created
3. **Photo Selection**: Your default browser opens with Google Photos
   - Browse and select the photos you want
   - Click "Open" or "Select" button
4. **Polling**: Script polls until selection is complete
5. **Retrieval**: Selected photos are listed with details
6. **Download**: Photos are automatically downloaded to local files

## What Gets Downloaded

- Photo files are saved as `photo_1_filename`, `photo_2_filename`, etc. (original format preserved)
- Original file names and formats are preserved (HEIC, PNG, JPG, etc.)
- Photos are downloaded at 800x800 pixels by default (customizable)

## Key Features

- **User Control**: Users explicitly select which photos to share
- **Programmatic Access**: After selection, you can download and process photos
- **Official API**: This is Google's recommended approach as of 2025
- **Token Caching**: OAuth token is cached for subsequent runs
- **Interactive**: Browser-based picker ensures good UX

## API Scopes

The script uses: `https://www.googleapis.com/auth/photospicker.mediaitems.readonly`

This scope allows:
- Creating picker sessions
- Polling session status
- Accessing selected media items
- Read-only access (no modifications)

## Available Size Parameters for Downloads

- `w640-h480` - Small (640x480)
- `w800-h800` - Medium (800x800, default)
- `w1200-h1200` - Large (1200x1200)
- `w1920-h1080` - Full HD (1920x1080)
- `w3840-h2160` - 4K (3840x2160)

To change the download size, modify the `download_photo()` call:

```python
picker.download_photo(base_url, filename, size='w1920-h1080')
```

## Files Generated

- `credentials.json` - OAuth credentials (from Google Cloud Console)
- `token.json` - Cached OAuth token (auto-generated)
- `photo_*` - Downloaded photos (preserves original format: HEIC, PNG, JPG, etc.)

## How It Works (Technical Details)

1. **Create Session**: `POST /sessions` creates a picker session and returns an `id`
2. **Get Picker URI**: Response includes a `pickerUri` for the browser
3. **User Selects Photos**: User browses Google Photos and makes selections
4. **Poll Status**: `GET /sessions/{id}` checks if `mediaItemsSet` is true
5. **Retrieve Media**: `GET /mediaItems?sessionId={id}` gets photo details with baseUrls
6. **Download**: Use the `baseUrl` from `mediaFile` to download the actual photo files

The `baseUrl` remains valid for 60 minutes after retrieval.

## Troubleshooting

### "Access blocked: PhotoLibrary has not completed the Google verification process"
- Go to **APIs & Services** > **OAuth consent screen**
- Under "Test users", add your Google account email
- Try again

### "credentials.json not found"
- Download OAuth credentials from Google Cloud Console
- Save as `credentials.json` in the script's directory

### "Invalid credentials" or UNAUTHENTICATED error
- Delete `token.json` to trigger fresh authentication
- Try again

### "Polling timeout reached"
- Make sure you actually selected photos in the browser
- Try again with a longer selection process
- The default timeout is 10 minutes

### Browser doesn't open automatically
- Copy the URL from the terminal and open it manually in your browser
- Complete the selection there
- Return to the terminal

### No photos download
- Check that you actually selected photos in the picker
- Verify the photos were listed in Step 5 output
- Check file permissions in your directory

## Security Considerations

- **Never commit** `credentials.json` or `token.json` to version control
- Add to `.gitignore`:
  ```
  credentials.json
  token.json
  photo_*
  ```
- Keep OAuth client secret private
- Use read-only scopes (no modification permissions)
- BaseURLs expire after 60 minutes for security

## Example: Accessing Selected Photos in Your Code

```python
picker = GooglePhotosPicker()
picker.authenticate()

# Create session
session = picker.create_picker_session()
session_id = session.get('id')  # Use 'id', not 'sessionId'
picker_uri = session.get('pickerUri')

# Open picker
webbrowser.open(picker_uri)

# Wait for selection (returns when user completes selection)
session_result = picker.poll_session_status(session_id)

# Get the selected photos
if session_result:
    media_items = picker.get_selected_media(session_id)
    for item in media_items:
        media_file = item.get('mediaFile', {})
        print(f"Selected: {media_file.get('filename')}")
        print(f"URL: {media_file.get('baseUrl')}")
        print(f"Type: {item.get('type')}")
```

## Comparing Picker vs. Library API

| Feature | Picker API | Library API |
|---------|-----------|-------------|
| Access user photos | ✅ Yes | ❌ No (as of Apr 2025) |
| UX | 📱 Browser picker | ⚙️ Programmatic |
| User control | ✅ Explicit selection | ❌ App-only access |
| Use case | General photo access | App-managed photos |
| Scope | `photospicker.mediaitems.readonly` | `photoslibrary.readonly.appcreateddata` |

## References

- [Google Photos Picker API Docs](https://developers.google.com/photos/picker/guides/get-started-picker)
- [API Reference](https://developers.google.com/photos/picker/reference/rest/v1)
- [OAuth Scopes](https://developers.google.com/photos/overview/authorization)
- [Google Auth Library for Python](https://google-auth.readthedocs.io/)
