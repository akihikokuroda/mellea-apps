#!/usr/bin/env python3
"""English-Japanese audio translator: Record → STT → Translate → TTS with voices."""

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
    import gtts
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False

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
    language: str = "en",
) -> str:
    """Transcribe audio using OpenAI's Whisper model (local).

    Args:
        audio_bytes: WAV audio as bytes
        whisper_model: Whisper model size (tiny, base, small, medium, large)
        language: Language code (e.g., "en" for English, "ja" for Japanese)

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

    print(f"Transcribing audio ({language})...")
    try:
        wav_file = io.BytesIO(audio_bytes)
        with wave.open(wav_file, 'rb') as wf:
            sample_rate = wf.getframerate()
            num_frames = wf.getnframes()
            audio_data = wf.readframes(num_frames)

        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        result = model.transcribe(audio_array, language=language, fp16=False)
        text = result.get("text", "").strip()
        if not text:
            print(f"Warning: Empty transcription from Whisper")
        print(f"Transcription: {text}")
        return text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        raise


async def translate_with_llm(
    text: str,
    source_lang: str = "English",
    target_lang: str = "Japanese",
    ollama_url: str = "http://localhost:11434",
    model_id: str = "granite4.1:3b",
) -> str:
    """Translate text using Mellea LLM via Ollama.

    Args:
        text: Text to translate
        source_lang: Source language name
        target_lang: Target language name
        ollama_url: Ollama server URL
        model_id: Model name in Ollama

    Returns:
        Translated text
    """
    backend = OllamaBackend(
        model_id=model_id,
        base_url=ollama_url,
    )

    ctx = SimpleContext()

    system_prompt = (
        f"You are a professional translator. Translate the following text from {source_lang} to {target_lang}. "
        f"Provide ONLY the translation, no explanations. Keep the tone and style of the original."
    )

    model_options = {
        ModelOption.TEMPERATURE: 0.3,
        ModelOption.SYSTEM_PROMPT: system_prompt,
    }

    print(f"Translating {source_lang} → {target_lang}...")
    print(f"Original: {text}")

    try:
        action = CBlock(text)
        mot, gen_ctx = await mfuncs.aact(
            action, ctx, backend, strategy=None, model_options=model_options
        )

        translation = await mot.avalue()
        translation = translation.strip()
        print(f"Translation: {translation}")
        return translation

    except Exception as e:
        print(f"Error calling translation LLM: {e}")
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
    language: str = "English",
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> bytes:
    """Synthesize text to speech using Edge TTS with retry logic.

    Args:
        text: Text to synthesize
        output_file: Optional output file path
        rate: Speech rate (e.g., "+0%", "+10%", "-10%")
        voice: Voice to use
        language: Language name for display
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        Audio bytes in MP3 format
    """
    import io

    print(f"Synthesizing {language} speech...")

    if not text or not text.strip():
        print("Warning: Empty text provided for TTS, returning empty bytes")
        return b""

    last_error = None
    for attempt in range(max_retries):
        try:
            communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)

            audio_data = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.write(chunk["data"])

            audio_bytes = audio_data.getvalue()

            if not audio_bytes:
                if attempt < max_retries - 1:
                    print(f"Warning: Empty audio data. Retrying {attempt + 1}/{max_retries - 1}...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    print("Error: Edge TTS returned no audio data after retries")
                    return b""

            if output_file:
                with open(output_file, 'wb') as f:
                    f.write(audio_bytes)
                print(f"Audio saved to: {output_file}")

            return audio_bytes

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"TTS error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}")
                print(f"Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
            else:
                print(f"Error synthesizing speech after {max_retries} attempts: {e}")
                print("\nTrying fallback TTS...")

                lang_code_map = {
                    "English": "en",
                    "Japanese": "ja",
                    "Spanish": "es",
                    "French": "fr",
                    "Chinese": "zh",
                }
                lang_code = lang_code_map.get(language, "en")

                fallback_audio = await synthesize_speech_fallback(
                    text=text,
                    output_file=output_file,
                    language_code=lang_code,
                    language=language,
                )

                if fallback_audio:
                    print("✓ Fallback TTS succeeded")
                    return fallback_audio
                else:
                    print("\nTroubleshooting:")
                    print("1. Check internet connection (Edge TTS requires Bing services)")
                    print("2. Verify the voice name is correct")
                    print("3. Try reducing text length")
                    print("4. Install fallback TTS: pip install gtts")
                    print("5. Wait a moment and retry (rate limiting may apply)")
                    raise


async def synthesize_speech_fallback(
    text: str,
    output_file: str | None = None,
    language_code: str = "en",
    language: str = "English",
) -> bytes:
    """Fallback TTS using gTTS when Edge TTS fails.

    Args:
        text: Text to synthesize
        output_file: Optional output file path
        language_code: Language code (e.g., "en", "ja")
        language: Language name for display

    Returns:
        Audio bytes in MP3 format
    """
    import io

    if not HAS_GTTS:
        print("Fallback TTS (gTTS) not installed. Install with: pip install gtts")
        return b""

    print(f"Using fallback TTS for {language}...")

    try:
        tts = gtts.gTTS(text=text, lang=language_code, slow=False)
        audio_data = io.BytesIO()
        tts.write_to_fp(audio_data)
        audio_bytes = audio_data.getvalue()

        if output_file:
            with open(output_file, 'wb') as f:
                f.write(audio_bytes)
            print(f"Audio saved to: {output_file}")

        return audio_bytes
    except Exception as e:
        print(f"Fallback TTS also failed: {e}")
        return b""


def play_audio(audio_bytes: bytes):
    """Play MP3 audio bytes through speakers.

    Args:
        audio_bytes: MP3 audio data as bytes
    """
    import io
    import subprocess
    import tempfile

    if not audio_bytes or len(audio_bytes) == 0:
        print("Warning: No audio to play (empty response)")
        return

    print("Playing audio...")
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        if sys.platform == "darwin":
            subprocess.run(["afplay", tmp_path], check=True)
        elif sys.platform == "linux":
            # subprocess.run(["aplay", tmp_path], check=True)
            subprocess.run(["mpg123", "-q", tmp_path], check=True)
        elif sys.platform == "win32":
            import winsound
            winsound.PlaySound(tmp_path, winsound.SND_FILENAME)

        print("Audio playback complete")
        Path(tmp_path).unlink()
    except Exception as e:
        print(f"Error playing audio: {e}")


async def process_single_translation(
    audio_bytes: bytes,
    source_lang: str = "en",
    source_lang_name: str = "English",
    target_lang: str = "ja",
    target_lang_name: str = "Japanese",
    source_voice: str = "en-US-AriaNeural",
    target_voice: str = "ja-JP-NanamiNeural",
    ollama_url: str = "http://localhost:11434",
    whisper_model: str = "base",
    llm_model: str = "granite4.1:3b",
    play_response: bool = True,
    turn_number: int = 1,
    output_dir: str | None = None,
) -> dict:
    """Process a single translation turn: STT → Translate → TTS (both languages).

    Args:
        audio_bytes: Input WAV audio as bytes
        source_lang: Source language code
        source_lang_name: Source language full name
        target_lang: Target language code
        target_lang_name: Target language full name
        source_voice: Voice for source language
        target_voice: Voice for target language
        ollama_url: Ollama server URL
        whisper_model: Whisper model size
        llm_model: LLM model name
        play_response: Whether to play audio responses
        turn_number: Turn number for display
        output_dir: Optional directory to save audio files

    Returns:
        Dict with original text, translation, and audio info
    """
    print("\n" + "=" * 60)
    print(f"TURN {turn_number} - Translation")
    print("=" * 60)

    # Step 1: STT (detect source language)
    print(f"\n[1/4] Speech-to-Text ({source_lang_name})")
    print("-" * 60)
    original_text = transcribe_with_whisper(audio_bytes, whisper_model, language=source_lang)

    if not original_text or len(original_text.strip()) < 2:
        print("Error: Could not transcribe audio. Please speak clearly and try again.")
        return {
            "original": original_text,
            "translation": None,
            "exit": False,
        }

    # Check for exit commands
    user_input_lower = original_text.lower().strip()
    if any(word in user_input_lower for word in ["bye", "quit", "exit", "goodbye", "さようなら", "やめる", "終了"]):
        return {
            "original": original_text,
            "translation": None,
            "exit": True,
        }

    # Step 2: Translate
    print(f"\n[2/4] Translation ({source_lang_name} → {target_lang_name})")
    print("-" * 60)
    translation = await translate_with_llm(
        original_text,
        source_lang=source_lang_name,
        target_lang=target_lang_name,
        ollama_url=ollama_url,
        model_id=llm_model,
    )

    if not translation or not translation.strip():
        print("Warning: Translation returned empty text")
        translation = "(Translation failed)"

    # Step 3: TTS for both languages
    print(f"\n[3/4] Text-to-Speech ({source_lang_name} & {target_lang_name})")
    print("-" * 60)

    original_audio = await synthesize_speech(
        original_text,
        rate="+0%",
        voice=source_voice,
        language=source_lang_name,
        max_retries=3,
        retry_delay=2.0,
    )

    translation_audio = await synthesize_speech(
        translation,
        rate="+0%",
        voice=target_voice,
        language=target_lang_name,
        max_retries=3,
        retry_delay=2.0,
    )

    # Save audio files if output directory specified
    audio_files = {}
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        original_file = output_path / f"turn_{turn_number:03d}_original_{source_lang}.mp3"
        translation_file = output_path / f"turn_{turn_number:03d}_translation_{target_lang}.mp3"

        with open(original_file, 'wb') as f:
            f.write(original_audio)
        with open(translation_file, 'wb') as f:
            f.write(translation_audio)

        audio_files = {
            "original": str(original_file),
            "translation": str(translation_file),
        }
        print(f"Audio files saved to {output_dir}")

    # Step 4: Play responses
    print(f"\n[4/4] Audio Playback")
    print("-" * 60)
    if play_response:
        print(f"Playing {source_lang_name}...")
        play_audio(original_audio)
        print(f"Playing {target_lang_name}...")
        play_audio(translation_audio)

    result = {
        "original": original_text,
        "translation": translation,
        "original_audio_bytes": len(original_audio),
        "translation_audio_bytes": len(translation_audio),
        "audio_files": audio_files,
        "exit": False,
    }

    print("\n" + "=" * 60)
    print("TURN RESULTS")
    print("=" * 60)
    print(f"{source_lang_name}: {result['original']}")
    print(f"{target_lang_name}: {result['translation']}")

    return result


async def interactive_translator(
    duration: int = 5,
    sample_rate: int = 16000,
    source_lang: str = "en",
    source_lang_name: str = "English",
    target_lang: str = "ja",
    target_lang_name: str = "Japanese",
    source_voice: str = "en-US-AriaNeural",
    target_voice: str = "ja-JP-NanamiNeural",
    ollama_url: str = "http://localhost:11434",
    whisper_model: str = "base",
    llm_model: str = "granite4.1:3b",
    play_response: bool = True,
    output_dir: str | None = None,
) -> None:
    """Interactive translation loop.

    Records audio, transcribes, translates, and plays both original and translation.
    Repeats until user says bye/quit/exit/goodbye/さようなら/終了.

    Args:
        duration: Recording duration in seconds per turn
        sample_rate: Sample rate in Hz
        source_lang: Source language code
        source_lang_name: Source language full name
        target_lang: Target language code
        target_lang_name: Target language full name
        source_voice: Voice for source language
        target_voice: Voice for target language
        ollama_url: Ollama server URL
        whisper_model: Whisper model size
        llm_model: LLM model name
        play_response: Whether to play audio responses
        output_dir: Optional directory to save audio files
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE TRANSLATOR")
    print("=" * 60)
    print(f"{source_lang_name} ↔ {target_lang_name}")
    print(f"Say 'bye', 'quit', 'exit', 'goodbye', 'さようなら', or '終了' to end")

    turn = 1
    while True:
        try:
            print(f"\n[Turn {turn}] Recording {source_lang_name}...")
            audio_data = record_audio(
                duration=duration,
                sample_rate=sample_rate,
                channels=1,
            )
            audio_bytes = audio_to_wav_bytes(audio_data, sample_rate)

            result = await process_single_translation(
                audio_bytes,
                source_lang=source_lang,
                source_lang_name=source_lang_name,
                target_lang=target_lang,
                target_lang_name=target_lang_name,
                source_voice=source_voice,
                target_voice=target_voice,
                ollama_url=ollama_url,
                whisper_model=whisper_model,
                llm_model=llm_model,
                play_response=play_response,
                turn_number=turn,
                output_dir=output_dir,
            )

            if result.get("exit"):
                print("\n" + "=" * 60)
                print("GOODBYE!")
                print("=" * 60)
                break

            turn += 1

        except KeyboardInterrupt:
            print("\n\nTranslation session interrupted.")
            break
        except Exception as e:
            print(f"\nError in turn {turn}: {e}")
            import traceback
            traceback.print_exc()
            continue


def main():
    parser = argparse.ArgumentParser(
        description="English-Japanese audio translator: Record → STT → Translate → TTS"
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
        "-s",
        "--save-input-audio",
        help="Save input audio to file",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Don't play response audio",
    )
    parser.add_argument(
        "--source-lang",
        default="en",
        help="Source language code (default: en)",
    )
    parser.add_argument(
        "--source-lang-name",
        default="English",
        help="Source language full name (default: English)",
    )
    parser.add_argument(
        "--target-lang",
        default="ja",
        help="Target language code (default: ja)",
    )
    parser.add_argument(
        "--target-lang-name",
        default="Japanese",
        help="Target language full name (default: Japanese)",
    )
    parser.add_argument(
        "--source-voice",
        default="en-US-AriaNeural",
        help="Source language voice (default: en-US-AriaNeural)",
    )
    parser.add_argument(
        "--target-voice",
        default="ja-JP-NanamiNeural",
        help="Target language voice (default: ja-JP-NanamiNeural)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive translation mode",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to save translation audio files",
    )

    args = parser.parse_args()

    try:
        if args.interactive:
            asyncio.run(
                interactive_translator(
                    duration=args.duration,
                    sample_rate=args.sample_rate,
                    source_lang=args.source_lang,
                    source_lang_name=args.source_lang_name,
                    target_lang=args.target_lang,
                    target_lang_name=args.target_lang_name,
                    source_voice=args.source_voice,
                    target_voice=args.target_voice,
                    ollama_url=args.ollama_url,
                    whisper_model=args.whisper_model,
                    llm_model=args.llm_model,
                    play_response=not args.no_play,
                    output_dir=args.output_dir,
                )
            )
            return

        # Single turn mode
        if args.audio_file:
            print(f"Loading audio from {args.audio_file}...")
            audio_data, sample_rate = sf.read(args.audio_file, dtype="int16")
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
            audio_bytes = audio_to_wav_bytes(audio_data, sample_rate)
        else:
            audio_data = record_audio(
                duration=args.duration,
                sample_rate=args.sample_rate,
                channels=1,
            )
            audio_bytes = audio_to_wav_bytes(audio_data, args.sample_rate)

        if args.save_input_audio:
            Path(args.save_input_audio).parent.mkdir(parents=True, exist_ok=True)
            sf.write(args.save_input_audio, audio_data, args.sample_rate)
            print(f"Input audio saved to: {args.save_input_audio}")

        asyncio.run(
            process_single_translation(
                audio_bytes,
                source_lang=args.source_lang,
                source_lang_name=args.source_lang_name,
                target_lang=args.target_lang,
                target_lang_name=args.target_lang_name,
                source_voice=args.source_voice,
                target_voice=args.target_voice,
                ollama_url=args.ollama_url,
                whisper_model=args.whisper_model,
                llm_model=args.llm_model,
                play_response=not args.no_play,
                turn_number=1,
                output_dir=args.output_dir,
            )
        )

    except Exception as e:
        print(f"\nTranslator pipeline failed: {e}")
        print("\nMake sure:")
        print("  1. Ollama is running: ollama serve")
        print("  2. LLM model is pulled: ollama pull granite4.1:3b")
        print("  3. Dependencies installed:")
        print("     pip install openai-whisper edge-tts")
        sys.exit(1)


if __name__ == "__main__":
    main()
