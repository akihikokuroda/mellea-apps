#!/usr/bin/env python3
"""Audio pipeline with Alexa integration: Record → STT → LLM → TTS (Alexa tool handling via prompting)."""

import argparse
import asyncio
import sys
import re
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
    from mellea.backends.ollama import OllamaModelBackend as OllamaBackend
    from mellea.backends.model_options import ModelOption
    from mellea.stdlib.components import CBlock
    from mellea.stdlib.context import SimpleContext
    import mellea.stdlib.functional as mfuncs
except ImportError:
    print("Error: Mellea not installed")
    print("Install with: pip install mellea[backends]")
    sys.exit(1)


# ============================================================================
# ALEXA TOOL IMPLEMENTATIONS
# ============================================================================

async def call_contact(contact: str) -> bytes:
    """Generate and synthesize voice command to call a contact via Alexa."""
    voice_command = f"Alexa, call {contact}"
    print(f"[ALEXA VOICE COMMAND] {voice_command}")
    audio_bytes = await synthesize_speech(voice_command, voice="en-US-AriaNeural")
    return audio_bytes


async def play_music(music: str) -> bytes:
    """Generate and synthesize voice command to play music via Alexa."""
    voice_command = f"Alexa, play {music}"
    print(f"[ALEXA VOICE COMMAND] {voice_command}")
    audio_bytes = await synthesize_speech(voice_command, voice="en-US-AriaNeural")
    return audio_bytes


async def get_weather(time_period: str) -> bytes:
    """Generate and synthesize voice command to get weather via Alexa."""
    voice_command = f"Alexa, what's the weather {time_period}"
    print(f"[ALEXA VOICE COMMAND] {voice_command}")
    audio_bytes = await synthesize_speech(voice_command, voice="en-US-AriaNeural")
    return audio_bytes


async def execute_alexa_command(command: str, args: str) -> bytes:
    """Execute an Alexa command and return synthesized voice command audio."""
    if command == "call_contact":
        return await call_contact(args)
    elif command == "play_music":
        return await play_music(args)
    elif command == "get_weather":
        return await get_weather(args)
    else:
        unknown_cmd = f"Unknown command: {command}"
        print(f"[ERROR] {unknown_cmd}")
        return await synthesize_speech(unknown_cmd)


# ============================================================================
# AUDIO PROCESSING FUNCTIONS
# ============================================================================

def record_audio(
    duration: int = 5,
    sample_rate: int = 16000,
    channels: int = 1,
) -> np.ndarray:
    """Record audio from microphone."""
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
    """Convert audio array to WAV bytes."""
    import io
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()


def transcribe_with_whisper(
    audio_bytes: bytes,
    whisper_model: str = "base",
) -> str:
    """Transcribe audio using OpenAI's Whisper model (local)."""
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
        wav_file = io.BytesIO(audio_bytes)
        with wave.open(wav_file, 'rb') as wf:
            sample_rate = wf.getframerate()
            num_frames = wf.getnframes()
            audio_data = wf.readframes(num_frames)

        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        audio_rms = np.sqrt(np.mean(audio_array ** 2))
        if audio_rms < 0.01:
            print(f"Warning: Audio appears to be silence or very quiet (RMS: {audio_rms:.6f})")

        result = model.transcribe(audio_array, language="en", fp16=False)
        text = result.get("text", "").strip()
        if not text:
            print("Warning: Empty transcription from Whisper")
        print(f"Transcription: {text}")
        return text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        raise


async def send_to_mellea_llm_with_alexa(
    user_text: str,
    ollama_url: str = "http://localhost:11434",
    model_id: str = "granite4.1:3b",
    system_prompt: str = "You are a helpful assistant.",
    enable_alexa_tools: bool = True,
) -> tuple[str, list[dict]]:
    """Send transcribed text to Mellea LLM with Alexa tools support.

    Args:
        user_text: User message
        ollama_url: Ollama server URL
        model_id: Model name in Ollama
        system_prompt: System prompt
        enable_alexa_tools: Whether to enable Alexa tools

    Returns:
        Tuple of (cleaned_response, list_of_alexa_commands)
    """
    backend = OllamaBackend(
        model_id=model_id,
        base_url=ollama_url,
    )

    ctx = SimpleContext()

    # Enhance system prompt with Alexa tool instructions
    enhanced_prompt = system_prompt
    if enable_alexa_tools:
        enhanced_prompt += (
            "\n\nYou have access to Alexa tools. Include Alexa commands in your response "
            "using the format [ALEXA: command_name(argument)]. ALWAYS include conversational text "
            "BEFORE or AFTER the Alexa commands. Do not respond ONLY with Alexa commands - always include spoken words.\n"
            "Available commands:\n"
            "- call_contact(name): Call someone (e.g., [ALEXA: call_contact(mom)])\n"
            "- play_music(song_or_artist): Play music (e.g., [ALEXA: play_music(jazz)])\n"
            "- get_weather(time_period): Get weather (e.g., [ALEXA: get_weather(tomorrow)])\n"
            "Example: I'll help you with that! [ALEXA: call_contact(john)]"
        )

    model_options = {
        ModelOption.TEMPERATURE: 0.7,
        ModelOption.SYSTEM_PROMPT: enhanced_prompt,
    }

    print(f"Sending to Mellea LLM via Ollama ({ollama_url})...")
    print(f"Model: {model_id}")
    print(f"User: {user_text}")

    try:
        action = CBlock(user_text)
        mot, gen_ctx = await mfuncs.aact(
            action, ctx, backend, strategy=None, model_options=model_options
        )

        raw_response = await mot.avalue()
        print(f"Assistant (raw): {raw_response}")

        # Parse out Alexa commands if enabled
        if enable_alexa_tools:
            cleaned_response, alexa_commands = parse_llm_response_for_alexa(raw_response)
            print(f"Assistant: {cleaned_response}")
            if alexa_commands:
                print(f"Alexa commands detected: {alexa_commands}")
            return cleaned_response, alexa_commands
        else:
            return raw_response, []

    except Exception as e:
        print(f"Error calling Mellea LLM: {e}")
        import traceback
        traceback.print_exc()
        print("Note: Make sure Ollama is running and model is pulled:")
        print(f"  ollama serve")
        print(f"  ollama pull {model_id}")
        raise


def parse_llm_response_for_alexa(response: str) -> tuple[str, list[dict]]:
    """Parse LLM response to extract Alexa commands.

    Looks for Alexa commands in format [ALEXA: command_name(argument)]

    Args:
        response: Raw LLM response text

    Returns:
        Tuple of (cleaned_response, list_of_alexa_commands)
    """
    alexa_pattern = r'\[ALEXA:\s*(\w+)\s*\(([^)]*)\)\s*\]'
    matches = re.findall(alexa_pattern, response)

    alexa_commands = []
    for match in matches:
        command_name = match[0].strip()
        argument = match[1].strip()
        alexa_commands.append({
            "command": command_name,
            "argument": argument,
        })

    cleaned_response = re.sub(alexa_pattern, '', response).strip()

    return cleaned_response, alexa_commands


async def execute_alexa_commands(alexa_commands: list[dict], play_alexa_commands: bool = True, wait_for_alexa_seconds: int = 20) -> dict:
    """Execute all Alexa commands from LLM response and play them as audio.

    Args:
        alexa_commands: List of Alexa command dicts
        play_alexa_commands: Whether to play the synthesized voice commands
        wait_for_alexa_seconds: Seconds to wait for Alexa to respond (default: 20)

    Returns:
        Dict with execution results
    """
    execution_results = {}

    for cmd in alexa_commands:
        command_name = cmd.get("command", "")
        argument = cmd.get("argument", "")
        try:
            audio_bytes = await execute_alexa_command(command_name, argument)
            execution_results[f"{command_name}({argument})"] = {
                "success": True,
                "audio_bytes": len(audio_bytes),
            }
            print(f"  ✓ Synthesized voice command ({len(audio_bytes)} bytes)")

            # Play the voice command to Alexa device
            if play_alexa_commands and audio_bytes:
                print(f"  🔊 Playing Alexa command to device...")
                play_audio(audio_bytes)

                # Wait for Alexa to respond
                print(f"  ⏳ Waiting {wait_for_alexa_seconds} seconds for Alexa to respond...")
                for i in range(wait_for_alexa_seconds):
                    await asyncio.sleep(1)
                    remaining = wait_for_alexa_seconds - i - 1
                    if remaining > 0 and remaining % 5 == 0:
                        print(f"     {remaining} seconds remaining...")
                print(f"  ✓ Alexa response time completed")

        except Exception as e:
            execution_results[f"{command_name}({argument})"] = {
                "success": False,
                "error": str(e),
            }
            print(f"  ✗ {command_name}({argument}): {e}")

    return execution_results


async def synthesize_speech(
    text: str,
    output_file: str | None = None,
    rate: str = "+0%",
    voice: str = "en-US-AriaNeural",
) -> bytes:
    """Synthesize text to speech using Edge TTS."""
    import io

    print("Synthesizing speech...")

    if not text or not text.strip():
        print("Warning: Empty text provided for TTS, returning empty bytes")
        return b""

    try:
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)

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
    """Play MP3 audio bytes through speakers."""
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
            subprocess.run(["aplay", tmp_path], check=True)
        elif sys.platform == "win32":
            import winsound
            winsound.PlaySound(tmp_path, winsound.SND_FILENAME)

        print("Audio playback complete")
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
    enable_alexa_tools: bool = True,
    wait_for_alexa_seconds: int = 20,
) -> dict:
    """Process a single conversation turn with Alexa tool support.

    STT → LLM (with Alexa tools) → Execute Alexa (+ wait) → TTS → Play

    Args:
        audio_bytes: Input WAV audio as bytes
        ollama_url: Ollama server URL
        whisper_model: Whisper model size
        llm_model: LLM model name
        system_prompt: System prompt for LLM
        play_response: Whether to play audio response
        turn_number: Turn number for display
        enable_alexa_tools: Whether to enable Alexa tools
        wait_for_alexa_seconds: Seconds to wait for Alexa response (default: 20)

    Returns:
        Dict with transcription, response, and exit flag
    """
    print("\n" + "=" * 60)
    print(f"TURN {turn_number}")
    print("=" * 60)

    # Step 1: STT
    print("\n[1/5] Speech-to-Text (Local Whisper)")
    print("-" * 60)
    transcription = transcribe_with_whisper(audio_bytes, whisper_model)

    if not transcription or len(transcription.strip()) < 2:
        print("Error: Could not transcribe audio. Please speak clearly and try again.")
        return {"transcription": transcription, "response": None, "exit": False}

    # Check for exit commands
    user_input_lower = transcription.lower().strip()
    if any(word in user_input_lower for word in ["bye", "quit", "exit", "goodbye"]):
        return {"transcription": transcription, "response": None, "exit": True}

    # Step 2: LLM with Alexa tools
    print("\n[2/5] Language Model Generation (with Alexa tools)")
    print("-" * 60)
    response, alexa_commands = await send_to_mellea_llm_with_alexa(
        transcription,
        ollama_url=ollama_url,
        model_id=llm_model,
        system_prompt=system_prompt,
        enable_alexa_tools=enable_alexa_tools,
    )

    if not response or not response.strip():
        print("Warning: LLM returned only Alexa commands with no text response")
        response = "(Executing Alexa commands)"

    # Step 3: Execute Alexa commands
    alexa_results = {}
    if enable_alexa_tools and alexa_commands:
        print("\n[3/5] Alexa Commands (Synthesize & Play Voice)")
        print("-" * 60)
        alexa_results = await execute_alexa_commands(
            alexa_commands,
            play_alexa_commands=True,
            wait_for_alexa_seconds=wait_for_alexa_seconds
        )

    # Step 4: TTS
    print("\n[4/5] Text-to-Speech (Edge TTS)")
    print("-" * 60)
    response_audio_bytes = await synthesize_speech(response)

    # Step 5: Play response
    print("\n[5/5] Audio Playback")
    print("-" * 60)
    if play_response:
        play_audio(response_audio_bytes)

    result = {
        "transcription": transcription,
        "response": response,
        "response_audio_bytes": len(response_audio_bytes),
        "alexa_commands": alexa_commands,
        "alexa_results": alexa_results,
        "exit": False,
    }

    print("\n" + "=" * 60)
    print("TURN RESULTS")
    print("=" * 60)
    print(f"User: {result['transcription']}")
    print(f"Response: {result['response']}")

    return result


async def interactive_audio_chat_with_alexa(
    duration: int = 5,
    sample_rate: int = 16000,
    ollama_url: str = "http://localhost:11434",
    whisper_model: str = "base",
    llm_model: str = "granite4.1:3b",
    system_prompt: str = "You are a helpful assistant.",
    play_response: bool = True,
    enable_alexa_tools: bool = True,
    wait_for_alexa_seconds: int = 20,
) -> None:
    """Interactive audio chat loop with Alexa tools.

    Records audio, transcribes, sends to LLM (which can call Alexa tools),
    synthesizes response, and plays it. Repeats until user says "bye", "quit", "exit", or "goodbye".

    Args:
        duration: Recording duration in seconds per turn
        sample_rate: Sample rate in Hz
        ollama_url: Ollama server URL
        whisper_model: Whisper model size
        llm_model: LLM model name
        system_prompt: System prompt for LLM
        play_response: Whether to play audio responses
        enable_alexa_tools: Whether to enable Alexa tools
        wait_for_alexa_seconds: Seconds to wait for Alexa response (default: 20)
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE AUDIO CHAT WITH ALEXA")
    print("=" * 60)
    print("Available Alexa tools (LLM decides when to use them):")
    print("  - call_contact(name): Call someone")
    print("  - play_music(song_or_artist): Play music")
    print("  - get_weather(time_period): Get weather forecast")
    print(f"Say 'bye', 'quit', 'exit', or 'goodbye' to end the conversation")

    turn = 1
    while True:
        try:
            print(f"\n[Turn {turn}] Recording...")
            audio_data = record_audio(
                duration=duration,
                sample_rate=sample_rate,
                channels=1,
            )
            audio_bytes = audio_to_wav_bytes(audio_data, sample_rate)

            result = await process_single_turn(
                audio_bytes,
                ollama_url=ollama_url,
                whisper_model=whisper_model,
                llm_model=llm_model,
                system_prompt=system_prompt,
                play_response=play_response,
                turn_number=turn,
                enable_alexa_tools=enable_alexa_tools,
                wait_for_alexa_seconds=wait_for_alexa_seconds,
            )

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
        description="Audio chat with Alexa integration"
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
        "--disable-alexa-tools",
        action="store_true",
        help="Disable Alexa tools",
    )
    parser.add_argument(
        "--alexa-wait",
        type=int,
        default=20,
        help="Seconds to wait for Alexa response (default: 20)",
    )

    args = parser.parse_args()

    try:
        # Interactive mode
        if args.interactive:
            asyncio.run(
                interactive_audio_chat_with_alexa(
                    duration=args.duration,
                    sample_rate=args.sample_rate,
                    ollama_url=args.ollama_url,
                    whisper_model=args.whisper_model,
                    llm_model=args.llm_model,
                    system_prompt=args.prompt,
                    play_response=not args.no_play,
                    enable_alexa_tools=not args.disable_alexa_tools,
                    wait_for_alexa_seconds=args.alexa_wait,
                )
            )
        else:
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
                process_single_turn(
                    audio_bytes,
                    ollama_url=args.ollama_url,
                    whisper_model=args.whisper_model,
                    llm_model=args.llm_model,
                    system_prompt=args.prompt,
                    play_response=not args.no_play,
                    turn_number=1,
                    enable_alexa_tools=not args.disable_alexa_tools,
                    wait_for_alexa_seconds=args.alexa_wait,
                )
            )

    except Exception as e:
        print(f"\nPipeline failed: {e}")
        print("\nMake sure:")
        print("  1. Ollama is running: ollama serve")
        print("  2. LLM model is pulled: ollama pull granite4.1:3b")
        print("  3. Dependencies installed:")
        print("     pip install openai-whisper edge-tts sounddevice soundfile numpy")
        sys.exit(1)


if __name__ == "__main__":
    main()
