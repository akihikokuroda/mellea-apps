#!/usr/bin/env python3
"""Full audio pipeline: Record → STT → LLM → TTS."""

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
    from pydub import AudioSegment
    from pydub.generators import Sine
except ImportError:
    print("Error: pydub not installed")
    print("Install with: pip install pydub")
    sys.exit(1)

try:
    import edge_tts
    import asyncio as aio
except ImportError:
    print("Error: edge-tts not installed")
    print("Install with: pip install edge-tts")
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

        result = model.transcribe(audio_array, language="en", fp16=False)
        text = result.get("text", "").strip()
        if not text:
            print("Warning: Empty transcription from Whisper")
        print(f"Transcription: {text}")
        return text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        raise


async def send_to_mellea_llm(
    user_text: str,
    ollama_url: str = "http://localhost:11434",
    model_id: str = "granite4.1:3b",
    system_prompt: str = "You are a helpful assistant.",
) -> str:
    """Send transcribed text to Mellea LLM for generation via Ollama.

    Args:
        user_text: User message
        ollama_url: Ollama server URL
        model_id: Model name in Ollama
        system_prompt: System prompt

    Returns:
        Generated response
    """
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

    try:
        # Create action block with user text
        action = CBlock(user_text)

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


async def full_audio_response_pipeline(
    audio_bytes: bytes,
    ollama_url: str = "http://localhost:11434",
    whisper_model: str = "base",
    llm_model: str = "granite4.1:3b",
    system_prompt: str = "You are a helpful assistant.",
    output_audio_file: str | None = None,
    play_response: bool = True,
) -> dict:
    """Full pipeline: Audio → STT → LLM → TTS → Play.

    Args:
        audio_bytes: Input WAV audio as bytes
        ollama_url: Ollama server URL
        whisper_model: Whisper model size
        llm_model: LLM model name
        system_prompt: System prompt for LLM
        output_audio_file: Optional file to save response audio
        play_response: Whether to play audio response

    Returns:
        Dict with transcription, response, and audio info
    """
    print("\n" + "=" * 60)
    print("FULL AUDIO RESPONSE PIPELINE")
    print("=" * 60)

    # Step 1: STT
    print("\n[1/4] Speech-to-Text (Local Whisper)")
    print("-" * 60)
    transcription = transcribe_with_whisper(audio_bytes, whisper_model)

    # Step 2: LLM
    print("\n[2/4] Language Model Generation (Mellea + Ollama)")
    print("-" * 60)
    response = await send_to_mellea_llm(
        transcription,
        ollama_url=ollama_url,
        model_id=llm_model,
        system_prompt=system_prompt,
    )

    # Step 3: TTS
    print("\n[3/4] Text-to-Speech (Edge TTS)")
    print("-" * 60)
    response_audio_bytes = await synthesize_speech(response, output_file=output_audio_file)

    # Step 4: Play response
    print("\n[4/4] Audio Playback")
    print("-" * 60)
    if play_response:
        play_audio(response_audio_bytes)

    result = {
        "transcription": transcription,
        "response": response,
        "response_audio_bytes": len(response_audio_bytes),
        "output_file": output_audio_file,
    }

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"User input: {result['transcription']}")
    print(f"LLM response: {result['response']}")
    print(f"Response audio: {result['response_audio_bytes']} bytes")
    if output_audio_file:
        print(f"Saved to: {result['output_file']}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Full audio pipeline: Record → STT → LLM → TTS → Playback"
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
        "--speech-rate",
        default="+0%",
        help="Speech rate: -50%% to +50%% (default: +0%%)",
    )
    parser.add_argument(
        "--voice",
        default="en-US-AriaNeural",
        help="Voice to use (default: en-US-AriaNeural)",
    )

    args = parser.parse_args()

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
    try:
        asyncio.run(
            full_audio_response_pipeline(
                audio_bytes,
                ollama_url=args.ollama_url,
                whisper_model=args.whisper_model,
                llm_model=args.llm_model,
                system_prompt=args.prompt,
                output_audio_file=args.output_audio,
                play_response=not args.no_play,
            )
        )
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        print("\nMake sure:")
        print("  1. Ollama is running: ollama serve")
        print("  2. LLM model is pulled: ollama pull granite4.1:3b")
        print("  3. Dependencies installed:")
        print("     pip install openai-whisper edge-tts")
        sys.exit(1)


if __name__ == "__main__":
    main()
