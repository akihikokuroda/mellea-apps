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

from dogmove import (
    WagStyle,
    WalkGait,
    EarMotion,
    tail_wag,
    head_tilt,
    walk_or_trot,
    paws_and_begging,
    ear_twitch,
    panting_interaction,
    begging,
    sit,
    MotionFrame,
)


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
    enable_dog_movements: bool = True,
) -> tuple[str, list[str]]:
    """Send transcribed text to Mellea LLM for generation via Ollama.

    Args:
        user_text: User message
        ollama_url: Ollama server URL
        model_id: Model name in Ollama
        system_prompt: System prompt
        enable_dog_movements: Whether to enable dog movement commands in LLM responses

    Returns:
        Tuple of (generated_response, movement_commands)
    """
    # Create Mellea backend for Ollama
    backend = OllamaBackend(
        model_id=model_id,
        base_url=ollama_url,
    )

    # Create simple context for async
    ctx = SimpleContext()

    # Enhance system prompt with dog movement instructions if enabled
    enhanced_prompt = system_prompt
    if enable_dog_movements:
        enhanced_prompt += (
            "\n\nYou can make a toy dog move by including movement commands in your response "
            "in the format [MOVE: command]. ALWAYS include conversational text BEFORE or AFTER the movement commands. "
            "Do not respond ONLY with movement commands - always include spoken words.\n"
            "Available commands:\n"
            "- tail_wag:style=fast_loose|slow_wide,duration=<ms>,swing_arc=<degrees>\n"
            "- head_tilt:angle=<degrees>,duration=<ms>\n"
            "- walk:gait=steady_walk|fast_trot,strides=<count>,stride_length=<mm>\n"
            "- beg:hip_angle=<degrees>,paw_lift=<mm>,hold_time=<ms>\n"
            "- ear_twitch:type=vertical|flaring,count=<cycles>,frequency=<hz>\n"
            "- pant:jaw_travel=<mm>,cycles=<count>,cycle_duration=<ms>\n"
            "- begging:rear_lift=<mm>,paw_lift=<mm>,hold_time=<ms>\n"
            "- sit:spine_angle=<degrees>,head_nod=<degrees>,sit_time=<ms>\n"
            "Example: I'm excited! [MOVE: tail_wag:style=fast_loose,duration=1000]"
        )

    # Model options with enhanced system prompt
    model_options = {
        ModelOption.TEMPERATURE: 0.7,
        ModelOption.SYSTEM_PROMPT: enhanced_prompt,
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
        raw_response = await mot.avalue()
        print(f"Assistant (raw): {raw_response}")

        # Parse out dog movements if enabled
        if enable_dog_movements:
            cleaned_response, movements = parse_llm_response_for_movements(raw_response)
            print(f"Assistant: {cleaned_response}")
            if movements:
                print(f"Dog movements detected: {movements}")
            return cleaned_response, movements
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

    # Handle empty text
    if not text or not text.strip():
        print("Warning: Empty text provided for TTS, returning empty bytes")
        return b""

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

    # Check if audio is empty
    if not audio_bytes or len(audio_bytes) == 0:
        print("Warning: No audio to play (empty response)")
        return

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


def execute_dog_movement(movement_command: str) -> list[MotionFrame]:
    """Execute a dog movement based on command string.

    Args:
        movement_command: Command in format "motion_type:param1=value1,param2=value2"
                         Example: "tail_wag:style=fast_loose,duration=1000"

    Returns:
        List of MotionFrame objects for the movement
    """
    print(f"Dog Movement: {movement_command}")

    parts = movement_command.split(":")
    if len(parts) < 1:
        print("Invalid movement command format")
        return []

    motion_type = parts[0].strip()
    params = {}

    if len(parts) > 1:
        param_string = parts[1]
        for param in param_string.split(","):
            if "=" in param:
                key, val = param.split("=", 1)
                params[key.strip()] = val.strip()

    try:
        if motion_type == "tail_wag":
            style = WagStyle(params.get("style", "fast_loose"))
            duration = float(params.get("duration", 1000.0))
            swing_arc = float(params.get("swing_arc", 45.0))
            return tail_wag(style=style, duration_ms=duration, swing_arc_degrees=swing_arc)

        elif motion_type == "head_tilt":
            angle = float(params.get("angle", 15.0))
            duration = float(params.get("duration", 300.0))
            return head_tilt(tilt_angle_degrees=angle, duration_ms=duration)

        elif motion_type == "walk":
            gait = WalkGait(params.get("gait", "steady_walk"))
            stride_count = int(params.get("strides", 4))
            stride_length = float(params.get("stride_length", 50.0))
            return walk_or_trot(gait=gait, stride_count=stride_count, stride_length_mm=stride_length)

        elif motion_type == "beg":
            hip_angle = float(params.get("hip_angle", 90.0))
            paw_lift = float(params.get("paw_lift", 30.0))
            hold_time = float(params.get("hold_time", 1000.0))
            return paws_and_begging(
                hip_hinge_angle_degrees=hip_angle,
                front_paw_lift_height_mm=paw_lift,
                hold_duration_ms=hold_time,
            )

        elif motion_type == "ear_twitch":
            ear_type = EarMotion(params.get("type", "vertical"))
            twitch_count = int(params.get("count", 3))
            frequency = float(params.get("frequency", 5.0))
            return ear_twitch(motion_type=ear_type, twitch_count=twitch_count, frequency_hz=frequency)

        elif motion_type == "pant":
            jaw_travel = float(params.get("jaw_travel", 15.0))
            cycles = int(params.get("cycles", 5))
            cycle_duration = float(params.get("cycle_duration", 200.0))
            return panting_interaction(
                jaw_travel_mm=jaw_travel,
                pant_cycles=cycles,
                cycle_duration_ms=cycle_duration,
            )

        elif motion_type == "begging":
            rear_lift = float(params.get("rear_lift", 40.0))
            paw_lift = float(params.get("paw_lift", 35.0))
            hold_time = float(params.get("hold_time", 1500.0))
            return begging(
                rear_leg_lift_height_mm=rear_lift,
                front_paw_lift_height_mm=paw_lift,
                hold_duration_ms=hold_time,
            )

        elif motion_type == "sit":
            spine_angle = float(params.get("spine_angle", 90.0))
            head_nod = float(params.get("head_nod", 15.0))
            sit_time = float(params.get("sit_time", 2000.0))
            return sit(
                spine_angle_degrees=spine_angle,
                head_nod_angle_degrees=head_nod,
                sit_duration_ms=sit_time,
            )

        else:
            print(f"Unknown movement type: {motion_type}")
            return []

    except Exception as e:
        print(f"Error executing dog movement: {e}")
        return []


def parse_llm_response_for_movements(response: str) -> tuple[str, list[str]]:
    """Parse LLM response to extract dog movement commands.

    Looks for movement commands in format [MOVE: command] or <<MOVE: command>>

    Args:
        response: Raw LLM response text

    Returns:
        Tuple of (cleaned_response, list_of_movement_commands)
    """
    import re

    movement_pattern = r'\[MOVE:\s*([^\]]+)\]|<<MOVE:\s*([^>]+)>>'
    matches = re.findall(movement_pattern, response)

    movement_commands = []
    for match in matches:
        command = match[0] if match[0] else match[1]
        movement_commands.append(command.strip())

    cleaned_response = re.sub(movement_pattern, '', response).strip()

    return cleaned_response, movement_commands


async def execute_llm_movements(movement_commands: list[str]) -> dict:
    """Execute all dog movements from LLM response.

    Args:
        movement_commands: List of movement command strings

    Returns:
        Dict with execution status and frame counts
    """
    execution_results = {}

    for cmd in movement_commands:
        frames = execute_dog_movement(cmd)
        execution_results[cmd] = {
            "frames": len(frames),
            "duration_ms": frames[-1].timestamp_ms if frames else 0,
        }

    return execution_results


async def process_single_turn(
    audio_bytes: bytes,
    ollama_url: str = "http://localhost:11434",
    whisper_model: str = "base",
    llm_model: str = "granite4.1:3b",
    system_prompt: str = "You are a helpful assistant.",
    play_response: bool = True,
    turn_number: int = 1,
    enable_dog_movements: bool = True,
) -> dict:
    """Process a single conversation turn: STT → LLM → Dog Movements → TTS → Play.

    Args:
        audio_bytes: Input WAV audio as bytes
        ollama_url: Ollama server URL
        whisper_model: Whisper model size
        llm_model: LLM model name
        system_prompt: System prompt for LLM
        play_response: Whether to play audio response
        turn_number: Turn number for display
        enable_dog_movements: Whether to enable dog movements from LLM

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

    # Check if transcription is empty
    if not transcription or len(transcription.strip()) < 2:
        print("Error: Could not transcribe audio. Please speak clearly and try again.")
        return {"transcription": transcription, "response": None, "exit": False}

    # Check for exit commands
    user_input_lower = transcription.lower().strip()
    if any(word in user_input_lower for word in ["bye", "quit", "exit", "goodbye"]):
        return {"transcription": transcription, "response": None, "exit": True}

    # Step 2: LLM
    print("\n[2/5] Language Model Generation (Mellea + Ollama)")
    print("-" * 60)
    response, movements = await send_to_mellea_llm(
        transcription,
        ollama_url=ollama_url,
        model_id=llm_model,
        system_prompt=system_prompt,
        enable_dog_movements=enable_dog_movements,
    )

    # Check if response is empty after movement extraction
    if not response or not response.strip():
        print("Warning: LLM returned only movement commands with no text response")
        response = "(Dog performed movements without speaking)"

    # Step 3: Execute dog movements
    movement_results = {}
    if enable_dog_movements and movements:
        print("\n[3/5] Dog Movements")
        print("-" * 60)
        movement_results = await execute_llm_movements(movements)
        for cmd, result in movement_results.items():
            print(f"  {cmd}: {result['frames']} frames, {result['duration_ms']}ms")

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
        "dog_movements": movements,
        "movement_results": movement_results,
        "exit": False,
    }

    print("\n" + "=" * 60)
    print("TURN RESULTS")
    print("=" * 60)
    print(f"User: {result['transcription']}")
    print(f"Assistant: {result['response']}")

    return result


async def interactive_audio_chat_with_dog(
    duration: int = 5,
    sample_rate: int = 16000,
    ollama_url: str = "http://localhost:11434",
    whisper_model: str = "base",
    llm_model: str = "granite4.1:3b",
    system_prompt: str = "You are a helpful assistant.",
    play_response: bool = True,
    enable_dog_movements: bool = True,
) -> None:
    """Interactive audio chat loop with dog movements.

    Records audio, transcribes, sends to LLM, executes dog movements,
    synthesizes response, and plays it. Repeats until user says bye/quit/exit/goodbye.

    Args:
        duration: Recording duration in seconds per turn
        sample_rate: Sample rate in Hz
        ollama_url: Ollama server URL
        whisper_model: Whisper model size
        llm_model: LLM model name
        system_prompt: System prompt for LLM
        play_response: Whether to play audio responses
        enable_dog_movements: Whether to enable dog movements from LLM
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE AUDIO CHAT WITH DOG MOVEMENTS")
    print("=" * 60)
    print(f"Dog Movements: {'Enabled' if enable_dog_movements else 'Disabled'}")
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
                enable_dog_movements=enable_dog_movements,
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


async def full_audio_response_pipeline(
    audio_bytes: bytes,
    ollama_url: str = "http://localhost:11434",
    whisper_model: str = "base",
    llm_model: str = "granite4.1:3b",
    system_prompt: str = "You are a helpful assistant.",
    output_audio_file: str | None = None,
    play_response: bool = True,
    enable_dog_movements: bool = True,
) -> dict:
    """Full pipeline: Audio → STT → LLM → TTS → Movements → Play.

    Args:
        audio_bytes: Input WAV audio as bytes
        ollama_url: Ollama server URL
        whisper_model: Whisper model size
        llm_model: LLM model name
        system_prompt: System prompt for LLM
        output_audio_file: Optional file to save response audio
        play_response: Whether to play audio response
        enable_dog_movements: Whether to enable dog movements from LLM

    Returns:
        Dict with transcription, response, audio info, and movements
    """
    print("\n" + "=" * 60)
    print("FULL AUDIO RESPONSE PIPELINE (WITH DOG MOVEMENTS)")
    print("=" * 60)

    # Step 1: STT
    print("\n[1/5] Speech-to-Text (Local Whisper)")
    print("-" * 60)
    transcription = transcribe_with_whisper(audio_bytes, whisper_model)

    # Step 2: LLM
    print("\n[2/5] Language Model Generation (Mellea + Ollama)")
    print("-" * 60)
    response, movements = await send_to_mellea_llm(
        transcription,
        ollama_url=ollama_url,
        model_id=llm_model,
        system_prompt=system_prompt,
        enable_dog_movements=enable_dog_movements,
    )

    # Check if response is empty after movement extraction
    if not response or not response.strip():
        print("Warning: LLM returned only movement commands with no text response")
        response = "(Dog performed movements without speaking)"

    # Step 3: Execute dog movements
    movement_results = {}
    if enable_dog_movements and movements:
        print("\n[3/5] Dog Movements")
        print("-" * 60)
        movement_results = await execute_llm_movements(movements)
        for cmd, result in movement_results.items():
            print(f"  {cmd}: {result['frames']} frames, {result['duration_ms']}ms")

    # Step 4: TTS
    print("\n[4/5] Text-to-Speech (Edge TTS)")
    print("-" * 60)
    response_audio_bytes = await synthesize_speech(response, output_file=output_audio_file)

    # Step 5: Play response
    print("\n[5/5] Audio Playback")
    print("-" * 60)
    if play_response:
        play_audio(response_audio_bytes)

    result = {
        "transcription": transcription,
        "response": response,
        "response_audio_bytes": len(response_audio_bytes),
        "output_file": output_audio_file,
        "dog_movements": movements,
        "movement_results": movement_results,
    }

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"User input: {result['transcription']}")
    print(f"LLM response: {result['response']}")
    print(f"Response audio: {result['response_audio_bytes']} bytes")
    if movements:
        print(f"Dog movements: {len(movements)} commands executed")
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
    parser.add_argument(
        "--enable-dog-movements",
        action="store_true",
        default=True,
        help="Enable dog movement commands from LLM (default: enabled)",
    )
    parser.add_argument(
        "--disable-dog-movements",
        action="store_true",
        help="Disable dog movement commands from LLM",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive chat mode (repeat until 'bye')",
    )

    args = parser.parse_args()
    enable_movements = not args.disable_dog_movements

    try:
        # Interactive mode
        if args.interactive:
            asyncio.run(
                interactive_audio_chat_with_dog(
                    duration=args.duration,
                    sample_rate=args.sample_rate,
                    ollama_url=args.ollama_url,
                    whisper_model=args.whisper_model,
                    llm_model=args.llm_model,
                    system_prompt=args.prompt,
                    play_response=not args.no_play,
                    enable_dog_movements=enable_movements,
                )
            )
            return

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
            full_audio_response_pipeline(
                audio_bytes,
                ollama_url=args.ollama_url,
                whisper_model=args.whisper_model,
                llm_model=args.llm_model,
                system_prompt=args.prompt,
                output_audio_file=args.output_audio,
                play_response=not args.no_play,
                enable_dog_movements=enable_movements,
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
