#!/usr/bin/env python3
"""Audio pipeline with DuckDuckGo search: Record → STT → LLM (with search) → TTS."""

import argparse
import asyncio
import sys
from pathlib import Path

try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    import whisper
except ImportError:
    print("Error: Required libraries not installed")
    print("Install with: pip install sounddevice soundfile numpy openai-whisper")
    sys.exit(1)

try:
    import edge_tts
    import asyncio as aio
except ImportError:
    print("Error: edge-tts not installed")
    print("Install with: pip install edge-tts")
    sys.exit(1)

try:
    from ddgs import DDGS
except ImportError:
    print("Error: ddgs not installed")
    print("Install with: pip install ddgs")
    sys.exit(1)

try:
    from mellea.backends.ollama import OllamaModelBackend as OllamaBackend
    from mellea.backends.model_options import ModelOption
    from mellea.stdlib.components import CBlock
    from mellea.stdlib.context import SimpleContext
    import mellea.stdlib.functional as mfuncs
except ImportError:
    print("Error: Mellea not installed")
    print("Install with: pip install mellea[backends]")
    sys.exit(1)


def record_audio(
    duration: int = 5,
    sample_rate: int = 16000,
    channels: int = 1,
) -> np.ndarray:
    """Record audio from microphone.

    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz
        channels: Number of channels

    Returns:
        NumPy array of audio data
    """
    print(f"Recording for {duration} seconds (press Ctrl+C to stop early)...")
    try:
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
        )
        sd.wait()
        print(f"Recording complete: {len(recording)} frames")
        return recording
    except KeyboardInterrupt:
        sd.stop()
        print("\nRecording stopped early")
        return recording


def audio_to_wav_bytes(audio_data: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Convert audio array to WAV bytes.

    Args:
        audio_data: NumPy array of audio
        sample_rate: Sample rate in Hz

    Returns:
        WAV file as bytes
    """
    import io

    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()


def transcribe_with_whisper(
    audio_bytes: bytes,
    whisper_model: str = "base",
) -> str:
    """Transcribe audio using OpenAI's Whisper model (local).

    Args:
        audio_bytes: WAV audio as bytes
        whisper_model: Whisper model size (tiny, base, small, medium, large)

    Returns:
        Transcribed text
    """
    import io
    import wave

    print(f"Loading Whisper model: {whisper_model}...")
    try:
        model = whisper.load_model(whisper_model)
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        raise

    print("Transcribing audio...")
    try:
        # Parse WAV bytes to extract raw audio data
        wav_file = io.BytesIO(audio_bytes)
        with wave.open(wav_file, 'rb') as wf:
            sample_rate = wf.getframerate()
            num_frames = wf.getnframes()
            audio_data = wf.readframes(num_frames)

        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Check if audio is mostly silence
        audio_rms = np.sqrt(np.mean(audio_array ** 2))
        if audio_rms < 0.01:
            print(f"Warning: Audio appears to be silence or very quiet (RMS: {audio_rms:.6f})")
            print("Tip: Speak clearly and closer to the microphone")

        result = model.transcribe(audio_array, language="en", fp16=False)
        text = result.get("text", "").strip()
        if not text:
            print("Warning: Empty transcription from Whisper")
            print("Tip: Audio may be too quiet or unclear")
        print(f"Transcription: {text}")
        return text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        raise


async def search_duckduckgo(query: str, max_results: int = 3, snippet_length: int = 2000, debug: bool = True) -> list[dict]:
    """Search using DuckDuckGo (ddgs library).

    Args:
        query: Search query
        max_results: Maximum number of results to return
        snippet_length: Maximum snippet length in characters
        debug: Enable debug logging

    Returns:
        List of search results with title, link, and snippet
    """
    # Skip search for empty or very short queries
    if not query or len(query.strip()) < 2:
        print("Skipping search: query too short or empty")
        return []

    query = query.strip()
    print(f"Searching DuckDuckGo for: '{query}'...")
    try:
        # Run search in thread to avoid blocking
        import concurrent.futures
        import threading
        import json

        results = []

        def perform_search():
            try:
                with DDGS(timeout=10) as ddgs:
                    search_results = list(ddgs.text(query, max_results=max_results + 2))
                    return search_results
            except Exception as e:
                print(f"  Error during search: {e}")
                return []

        # Run in thread pool
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            search_results = await loop.run_in_executor(executor, perform_search)

        # Debug: Print raw results
        if debug:
            print(f"\n  [DEBUG] Raw search results count: {len(search_results)}")
            if search_results:
                print(f"  [DEBUG] First result keys: {list(search_results[0].keys()) if search_results else 'N/A'}")
                print(f"  [DEBUG] Full raw results:")
                for i, result in enumerate(search_results[:max_results], 1):
                    print(f"    Result {i}:")
                    for key, value in result.items():
                        if isinstance(value, str) and len(value) > 100:
                            print(f"      {key}: {value[:100]}...")
                        else:
                            print(f"      {key}: {value}")
            print()

        if search_results:
            for result in search_results:
                # Debug: Check which results have required fields
                has_title = bool(result.get("title"))
                has_body = bool(result.get("body"))

                if debug:
                    print(f"  [DEBUG] Result - has_title: {has_title}, has_body: {has_body}, "
                          f"title: '{result.get('title', '')[:30]}...', "
                          f"body: '{result.get('body', '')[:30]}...'")

                if has_title and has_body:
                    results.append({
                        "title": result.get("title", "")[:150],
                        "snippet": result.get("body", "")[:snippet_length],
                        "link": result.get("href", ""),
                    })

        if results:
            print(f"  → Found {len(results)} search result(s)")
            for i, r in enumerate(results[:max_results], 1):
                print(f"     {i}. {r['title'][:50]}...")
        else:
            print(f"  → No results found from search")
            if search_results:
                print(f"  → (Note: Got {len(search_results)} results but none had both title and body)")

        return results[:max_results]

    except Exception as e:
        print(f"Warning: Error searching DuckDuckGo: {e}")
        import traceback
        traceback.print_exc()
        return []


def format_search_context(search_results: list[dict]) -> str:
    """Format search results into context string.

    Args:
        search_results: List of search results

    Returns:
        Formatted context string
    """
    if not search_results:
        return ""

    context = "\n\n[CONTEXT FROM WEB SEARCH]\n"
    for i, result in enumerate(search_results, 1):
        title = result.get('title', '').strip()
        snippet = result.get('snippet', '').strip()
        link = result.get('link', '').strip()

        if title or snippet:
            context += f"\n{i}. {title}\n"
            if snippet:
                context += f"   {snippet}\n"
            if link:
                context += f"   Source: {link}"

    context += "\n[END CONTEXT]\n"
    return context


async def send_to_mellea_llm_with_search(
    user_text: str,
    ollama_url: str = "http://localhost:11434",
    model_id: str = "granite4.1:3b",
    system_prompt: str = "You are a helpful assistant.",
    enable_search: bool = True,
    snippet_length: int = 2000,
) -> str:
    """Send transcribed text to Mellea LLM with optional web search context.

    Args:
        user_text: User message
        ollama_url: Ollama server URL
        model_id: Model name in Ollama
        system_prompt: System prompt
        enable_search: Whether to search DuckDuckGo for context
        snippet_length: Maximum snippet length in characters

    Returns:
        Generated response
    """
    # Perform search if enabled
    search_context = ""
    if enable_search:
        search_results = await search_duckduckgo(user_text, max_results=3, snippet_length=snippet_length)
        if search_results:
            search_context = format_search_context(search_results)
            print(f"Found {len(search_results)} search results")

    # Create Mellea backend for Ollama
    backend = OllamaBackend(
        model_id=model_id,
        base_url=ollama_url,
    )

    # Create simple context for async
    ctx = SimpleContext()

    # Model options with system prompt
    model_options = {
        ModelOption.TEMPERATURE: 0.7,
        ModelOption.SYSTEM_PROMPT: system_prompt,
    }

    print(f"Sending to Mellea LLM via Ollama ({ollama_url})...")
    print(f"Model: {model_id}")
    print(f"User: {user_text}")

    # Debug: Show what's being sent to LLM
    if search_context:
        print(f"\n[DEBUG] Search context being sent to LLM:")
        print(f"{search_context}")
    else:
        print(f"\n[DEBUG] No search context (search disabled or no results)")

    try:
        # Create action block with user text and search context
        action_text = user_text + search_context
        print(f"\n{'='*60}")
        print(f"[DEBUG] FULL PROMPT BEING SENT TO LLM")
        print(f"{'='*60}")
        print(f"Combined prompt length: {len(action_text)} characters")
        print(f"{'='*60}")
        print(action_text)
        print(f"{'='*60}\n")

        action = CBlock(action_text)

        # Generate response via Mellea with Ollama backend
        # aact returns (ModelOutputThunk, Context)
        mot, gen_ctx = await mfuncs.aact(
            action, ctx, backend, strategy=None, model_options=model_options
        )

        # Resolve the thunk to get the actual string
        response = await mot.avalue()
        print(f"Assistant: {response}")
        return response
    except Exception as e:
        print(f"Error calling Mellea LLM: {e}")
        import traceback
        traceback.print_exc()
        print("Note: Make sure Ollama is running and model is pulled:")
        print(f"  ollama serve")
        print(f"  ollama pull {model_id}")
        raise


async def synthesize_speech(
    text: str,
    output_file: str | None = None,
    rate: str = "+0%",
    voice: str = "en-US-AriaNeural",
) -> bytes:
    """Synthesize text to speech using Edge TTS.

    Args:
        text: Text to synthesize
        output_file: Optional output file path
        rate: Speech rate (e.g., "+0%", "+10%", "-10%")
        voice: Voice to use (default: en-US-AriaNeural)

    Returns:
        Audio bytes in MP3 format
    """
    import io

    print("Synthesizing speech...")
    try:
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)

        # Collect audio chunks
        audio_data = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])

        audio_bytes = audio_data.getvalue()

        if output_file:
            with open(output_file, 'wb') as f:
                f.write(audio_bytes)
            print(f"Audio saved to: {output_file}")

        return audio_bytes
    except Exception as e:
        print(f"Error synthesizing speech: {e}")
        raise


def play_audio(audio_bytes: bytes):
    """Play MP3 audio bytes through speakers.

    Args:
        audio_bytes: MP3 audio data as bytes
    """
    import io
    import subprocess
    import tempfile

    print("Playing audio...")
    try:
        # Save to temporary file and play with system audio
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Use system player
        if sys.platform == "darwin":  # macOS
            subprocess.run(["afplay", tmp_path], check=True)
        elif sys.platform == "linux":
            subprocess.run(["aplay", tmp_path], check=True)
        elif sys.platform == "win32":
            import winsound
            winsound.PlaySound(tmp_path, winsound.SND_FILENAME)

        print("Audio playback complete")

        # Clean up
        Path(tmp_path).unlink()
    except Exception as e:
        print(f"Error playing audio: {e}")


async def process_single_turn(
    audio_bytes: bytes,
    ollama_url: str = "http://localhost:11434",
    whisper_model: str = "base",
    llm_model: str = "granite4.1:3b",
    system_prompt: str = "You are a helpful assistant.",
    play_response: bool = True,
    turn_number: int = 1,
    enable_search: bool = True,
    snippet_length: int = 2000,
) -> dict:
    """Process a single conversation turn: STT → LLM (with search) → TTS → Play.

    Args:
        audio_bytes: Input WAV audio as bytes
        ollama_url: Ollama server URL
        whisper_model: Whisper model size
        llm_model: LLM model name
        system_prompt: System prompt for LLM
        play_response: Whether to play audio response
        turn_number: Turn number for display
        enable_search: Whether to enable DuckDuckGo search
        snippet_length: Maximum snippet length in characters

    Returns:
        Dict with transcription and response
    """
    print("\n" + "=" * 60)
    print(f"TURN {turn_number}")
    print("=" * 60)

    # Step 1: STT
    print("\n[1/4] Speech-to-Text (Local Whisper)")
    print("-" * 60)
    transcription = transcribe_with_whisper(audio_bytes, whisper_model)

    # Check if transcription is empty
    if not transcription or len(transcription.strip()) < 2:
        print("Error: Could not transcribe audio. Please speak clearly and try again.")
        return {"transcription": transcription, "response": None, "exit": False}

    # Check for exit commands
    user_input_lower = transcription.lower().strip()
    if any(word in user_input_lower for word in ["bye", "quit", "exit", "goodbye"]):
        return {"transcription": transcription, "response": None, "exit": True}

    # Step 2: LLM with Search
    print("\n[2/4] Language Model Generation (with DuckDuckGo Search)")
    print("-" * 60)
    response = await send_to_mellea_llm_with_search(
        transcription,
        ollama_url=ollama_url,
        model_id=llm_model,
        system_prompt=system_prompt,
        enable_search=enable_search,
        snippet_length=snippet_length,
    )

    # Step 3: TTS
    print("\n[3/4] Text-to-Speech (Edge TTS)")
    print("-" * 60)
    response_audio_bytes = await synthesize_speech(response)

    # Step 4: Play response
    print("\n[4/4] Audio Playback")
    print("-" * 60)
    if play_response:
        play_audio(response_audio_bytes)

    result = {
        "transcription": transcription,
        "response": response,
        "response_audio_bytes": len(response_audio_bytes),
        "exit": False,
    }

    print("\n" + "=" * 60)
    print("TURN RESULTS")
    print("=" * 60)
    print(f"User: {result['transcription']}")
    print(f"Assistant: {result['response']}")

    return result


async def interactive_audio_chat_with_search(
    duration: int = 5,
    sample_rate: int = 16000,
    ollama_url: str = "http://localhost:11434",
    whisper_model: str = "base",
    llm_model: str = "granite4.1:3b",
    system_prompt: str = "You are a helpful assistant.",
    play_response: bool = True,
    enable_search: bool = True,
    snippet_length: int = 2000,
) -> None:
    """Interactive audio chat loop with web search capabilities.

    Records audio, transcribes, searches the web, sends to LLM, synthesizes response, and plays it.
    Repeats until user says "bye", "quit", "exit", or "goodbye".

    Args:
        duration: Recording duration in seconds per turn
        sample_rate: Sample rate in Hz
        ollama_url: Ollama server URL
        whisper_model: Whisper model size
        llm_model: LLM model name
        system_prompt: System prompt for LLM
        play_response: Whether to play audio responses
        enable_search: Whether to enable DuckDuckGo search
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE AUDIO CHAT WITH SEARCH")
    print("=" * 60)
    print(f"Web Search: {'Enabled' if enable_search else 'Disabled'}")
    print(f"Say 'bye', 'quit', 'exit', or 'goodbye' to end the conversation")

    turn = 1
    while True:
        try:
            # Record audio
            print(f"\n[Turn {turn}] Recording...")
            audio_data = record_audio(
                duration=duration,
                sample_rate=sample_rate,
                channels=1,
            )
            audio_bytes = audio_to_wav_bytes(audio_data, sample_rate)

            # Process turn
            result = await process_single_turn(
                audio_bytes,
                ollama_url=ollama_url,
                whisper_model=whisper_model,
                llm_model=llm_model,
                system_prompt=system_prompt,
                play_response=play_response,
                turn_number=turn,
                enable_search=enable_search,
                snippet_length=snippet_length,
            )

            # Check if user wants to exit
            if result.get("exit"):
                print("\n" + "=" * 60)
                print("GOODBYE!")
                print("=" * 60)
                break

            turn += 1

        except KeyboardInterrupt:
            print("\n\nConversation interrupted.")
            break
        except Exception as e:
            print(f"\nError in turn {turn}: {e}")
            import traceback
            traceback.print_exc()
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Audio chat with DuckDuckGo web search integration"
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=5,
        help="Recording duration in seconds (default: 5)",
    )
    parser.add_argument(
        "-r",
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "-f",
        "--audio-file",
        help="Path to WAV file to process instead of recording",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model size: tiny, base, small, medium, large (default: base)",
    )
    parser.add_argument(
        "--llm-model",
        default="granite4.1:3b",
        help="Ollama LLM model name (default: granite4.1:3b)",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        default="You are a helpful assistant.",
        help="System prompt for LLM",
    )
    parser.add_argument(
        "-s",
        "--save-input-audio",
        help="Save input audio to file",
    )
    parser.add_argument(
        "-o",
        "--output-audio",
        help="Save response audio to file",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Don't play response audio",
    )
    parser.add_argument(
        "--voice",
        default="en-US-AriaNeural",
        help="Voice to use (default: en-US-AriaNeural)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive chat mode (repeat until 'bye')",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Disable DuckDuckGo web search",
    )
    parser.add_argument(
        "--snippet-length",
        type=int,
        default=2000,
        help="Maximum snippet length in characters (default: 2000)",
    )

    args = parser.parse_args()

    try:
        # Interactive mode
        if args.interactive:
            asyncio.run(
                interactive_audio_chat_with_search(
                    duration=args.duration,
                    sample_rate=args.sample_rate,
                    ollama_url=args.ollama_url,
                    whisper_model=args.whisper_model,
                    llm_model=args.llm_model,
                    system_prompt=args.prompt,
                    play_response=not args.no_play,
                    enable_search=not args.no_search,
                    snippet_length=args.snippet_length,
                )
            )
        else:
            # Single turn mode
            # Get input audio data
            if args.audio_file:
                print(f"Loading audio from {args.audio_file}...")
                audio_data, sample_rate = sf.read(args.audio_file, dtype="int16")
                if len(audio_data.shape) > 1:
                    audio_data = audio_data[:, 0]  # Convert stereo to mono
                audio_bytes = audio_to_wav_bytes(audio_data, sample_rate)
            else:
                # Record from microphone
                audio_data = record_audio(
                    duration=args.duration,
                    sample_rate=args.sample_rate,
                    channels=1,
                )
                audio_bytes = audio_to_wav_bytes(audio_data, args.sample_rate)

            # Save input audio if requested
            if args.save_input_audio:
                Path(args.save_input_audio).parent.mkdir(parents=True, exist_ok=True)
                sf.write(args.save_input_audio, audio_data, args.sample_rate)
                print(f"Input audio saved to: {args.save_input_audio}")

            # Process through pipeline
            asyncio.run(
                process_single_turn(
                    audio_bytes,
                    ollama_url=args.ollama_url,
                    whisper_model=args.whisper_model,
                    llm_model=args.llm_model,
                    system_prompt=args.prompt,
                    play_response=not args.no_play,
                    turn_number=1,
                    enable_search=not args.no_search,
                    snippet_length=args.snippet_length,
                )
            )

    except Exception as e:
        print(f"\nPipeline failed: {e}")
        print("\nMake sure:")
        print("  1. Ollama is running: ollama serve")
        print("  2. LLM model is pulled: ollama pull granite4.1:3b")
        print("  3. Dependencies installed:")
        print("     pip install openai-whisper edge-tts ddgs")
        sys.exit(1)


if __name__ == "__main__":
    main()
