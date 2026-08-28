#!/usr/bin/env python3
"""Audio trivia quiz: Generate trivia question → Record answer → STT → LLM verification → TTS feedback."""

import argparse
import asyncio
import sys
import json
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


async def generate_trivia_question(
    category: str = "general knowledge",
    difficulty: str = "medium",
    ollama_url: str = "http://localhost:11434",
    model_id: str = "granite4.1:3b",
    round_number: int = 1,
) -> dict:
    """Generate a trivia question using Mellea LLM.

    Args:
        category: Trivia category (general knowledge, history, science, etc.)
        difficulty: Difficulty level (easy, medium, hard)
        ollama_url: Ollama server URL
        model_id: Model name in Ollama
        round_number: Round number to help with variety

    Returns:
        Dict with question, correct_answer, and explanation
    """
    backend = OllamaBackend(
        model_id=model_id,
        base_url=ollama_url,
    )

    ctx = SimpleContext()

    system_prompt = """You are a trivia question generator. Generate ONE unique trivia question.
Output ONLY valid JSON, nothing else. Use this exact format:
{
    "question": "The question text",
    "correct_answer": "The correct answer",
    "alternatives": ["wrong answer 1", "wrong answer 2"],
    "explanation": "Why this is correct"
}"""

    model_options = {
        ModelOption.TEMPERATURE: 0.9,
        ModelOption.SYSTEM_PROMPT: system_prompt,
    }

    # Add variety hint for math to avoid repetition
    variety_hint = ""
    if category.lower() == "math" or "math" in category.lower():
        variety_hint = f" Make it different from previous questions. Question #{round_number}."

    prompt = f"Generate a {difficulty} {category} trivia question.{variety_hint}"

    print(f"Generating trivia question ({difficulty} - {category})...")
    print(f"Sending to Mellea LLM via Ollama ({ollama_url})...")
    print(f"Model: {model_id}")

    try:
        action = CBlock(prompt)

        mot, gen_ctx = await mfuncs.aact(
            action, ctx, backend, strategy=None, model_options=model_options
        )

        response = await mot.avalue()
        print(f"Raw response length: {len(response)} chars")

        # Try to extract JSON from response
        response_text = response.strip()

        # Find JSON in response (it might have extra text)
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            print(f"Extracted JSON: {json_str[:100]}...")
            trivia_data = json.loads(json_str)

            # Validate required fields
            if all(key in trivia_data for key in ["question", "correct_answer", "explanation"]):
                print(f"✓ Generated trivia question: {trivia_data['question'][:50]}...")
                return trivia_data

        print(f"Warning: Could not extract valid JSON from response")
        print(f"Full response: {response}")
        raise ValueError("Invalid JSON response")

    except Exception as e:
        print(f"Error generating trivia question: {e}")
        print(f"Retrying with simpler prompt...")

        # Fallback: try simpler prompt
        try:
            simple_prompt = f"Generate a {category} {difficulty} trivia question in JSON format only."
            action = CBlock(simple_prompt)
            mot, gen_ctx = await mfuncs.aact(
                action, ctx, backend, strategy=None, model_options=model_options
            )
            response = await mot.avalue()

            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                trivia_data = json.loads(json_str)
                if all(key in trivia_data for key in ["question", "correct_answer", "explanation"]):
                    print(f"✓ Generated trivia question (retry): {trivia_data['question'][:50]}...")
                    return trivia_data
        except:
            pass

        # Last resort: use hardcoded questions with variety
        questions_bank = [
            {
                "question": "What is 7 + 8?",
                "correct_answer": "15",
                "alternatives": ["14", "16"],
                "explanation": "Seven plus eight equals fifteen."
            },
            {
                "question": "What is 12 * 5?",
                "correct_answer": "60",
                "alternatives": ["50", "70"],
                "explanation": "Twelve times five equals sixty."
            },
            {
                "question": "What is 100 divided by 4?",
                "correct_answer": "25",
                "alternatives": ["20", "30"],
                "explanation": "One hundred divided by four equals twenty-five."
            },
            {
                "question": "What is the square of 9?",
                "correct_answer": "81",
                "alternatives": ["72", "90"],
                "explanation": "Nine squared (9 × 9) equals eighty-one."
            },
            {
                "question": "What is 15 - 7?",
                "correct_answer": "8",
                "alternatives": ["7", "9"],
                "explanation": "Fifteen minus seven equals eight."
            },
        ]

        # Select based on round number to avoid repetition
        selected = questions_bank[(round_number - 1) % len(questions_bank)]
        print(f"⚠ Using fallback question: {selected['question']}")
        return selected


async def verify_answer(
    user_answer: str,
    correct_answer: str,
    question: str,
    explanation: str,
    ollama_url: str = "http://localhost:11434",
    model_id: str = "granite4.1:3b",
) -> dict:
    """Verify user answer using Mellea LLM.

    Args:
        user_answer: User's spoken answer
        correct_answer: Correct answer
        question: The trivia question
        explanation: Explanation of correct answer
        ollama_url: Ollama server URL
        model_id: Model name in Ollama

    Returns:
        Dict with is_correct, explanation, feedback
    """
    backend = OllamaBackend(
        model_id=model_id,
        base_url=ollama_url,
    )

    ctx = SimpleContext()

    system_prompt = """You are a trivia quiz judge. Determine if the user's answer is correct.
Return a JSON response with these exact fields:
{
    "is_correct": true or false,
    "explanation": "Why this answer is correct or incorrect",
    "feedback": "Encouraging or constructive feedback for the user"
}

Be lenient with spelling and minor variations. Return ONLY JSON, no other text."""

    model_options = {
        ModelOption.TEMPERATURE: 0.5,
        ModelOption.SYSTEM_PROMPT: system_prompt,
    }

    prompt = f"""Question: {question}

Correct answer: {correct_answer}

User answered: {user_answer}

Explanation of correct answer: {explanation}

Is the user's answer correct? Respond with JSON only."""

    print(f"Verifying answer...")
    print(f"User said: '{user_answer}'")
    print(f"Correct answer: '{correct_answer}'")

    try:
        action = CBlock(prompt)

        mot, gen_ctx = await mfuncs.aact(
            action, ctx, backend, strategy=None, model_options=model_options
        )

        response = await mot.avalue()

        try:
            verdict = json.loads(response)
            print(f"✓ Answer verification: {'CORRECT' if verdict.get('is_correct') else 'INCORRECT'}")
            return verdict
        except json.JSONDecodeError:
            print(f"Warning: Could not parse verification JSON")
            print(f"Response: {response}")
            # Fallback: simple string matching
            is_correct = user_answer.lower().strip() == correct_answer.lower().strip()
            return {
                "is_correct": is_correct,
                "explanation": explanation,
                "feedback": "Good try!" if is_correct else "Not quite, try again next time!"
            }

    except Exception as e:
        print(f"Error verifying answer: {e}")
        import traceback
        traceback.print_exc()
        raise


async def synthesize_speech(
    text: str,
    output_file: str | None = None,
    rate: str = "+0%",
    voice: str = "en-US-AriaNeural",
) -> bytes:
    """Synthesize text to speech using Edge TTS."""
    import io

    print("Synthesizing speech...")
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
    import io
    import subprocess
    import tempfile

    print("Playing audio...")
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        if sys.platform == "darwin":  # macOS
            subprocess.run(["afplay", tmp_path], check=True)
        elif sys.platform == "linux":
            subprocess.run(["mpg123", "-q", tmp_path], check=True)
        elif sys.platform == "win32":
            import winsound
            winsound.PlaySound(tmp_path, winsound.SND_FILENAME)

        print("Audio playback complete")

        Path(tmp_path).unlink()
    except Exception as e:
        print(f"Error playing audio: {e}")


async def run_trivia_round(
    duration: int = 5,
    sample_rate: int = 16000,
    category: str = "general knowledge",
    difficulty: str = "medium",
    ollama_url: str = "http://localhost:11434",
    whisper_model: str = "base",
    llm_model: str = "granite4.1:3b",
    play_response: bool = True,
    round_number: int = 1,
) -> dict:
    """Run a single trivia round: Generate question → Ask → Record answer → Verify → Feedback.

    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz
        category: Trivia category
        difficulty: Question difficulty
        ollama_url: Ollama server URL
        whisper_model: Whisper model size
        llm_model: LLM model name
        play_response: Whether to play audio responses
        round_number: Round number for display

    Returns:
        Dict with question, user_answer, is_correct, and other metadata
    """
    print("\n" + "=" * 60)
    print(f"TRIVIA ROUND {round_number}")
    print("=" * 60)

    # Step 1: Generate trivia question
    print("\n[1/4] Generating Trivia Question")
    print("-" * 60)
    trivia_data = await generate_trivia_question(
        category=category,
        difficulty=difficulty,
        ollama_url=ollama_url,
        model_id=llm_model,
        round_number=round_number,
    )

    question = trivia_data.get("question", "")
    correct_answer = trivia_data.get("correct_answer", "")
    explanation = trivia_data.get("explanation", "")

    print(f"Q: {question}")
    print(f"   (Hint: Think about {category})")

    # Step 2: Synthesize and play question
    print("\n[2/4] Text-to-Speech Question")
    print("-" * 60)
    question_audio = await synthesize_speech(f"Question: {question}")
    if play_response:
        play_audio(question_audio)

    # Step 3: Record user answer
    print("\n[3/4] Recording Answer")
    print("-" * 60)
    print("Please answer the question...")
    audio_data = record_audio(duration=duration, sample_rate=sample_rate, channels=1)
    audio_bytes = audio_to_wav_bytes(audio_data, sample_rate)

    # Step 4: Transcribe answer
    print("\n[4/4] Speech-to-Text Answer")
    print("-" * 60)
    user_answer = transcribe_with_whisper(audio_bytes, whisper_model)

    if not user_answer or len(user_answer.strip()) < 1:
        print("Error: Could not transcribe answer. Please speak clearly and try again.")
        return {
            "question": question,
            "user_answer": None,
            "is_correct": False,
            "verification": None,
            "skip": True,
        }

    # Step 5: Verify answer
    print("\n[5/5] Verifying Answer")
    print("-" * 60)
    verdict = await verify_answer(
        user_answer,
        correct_answer,
        question,
        explanation,
        ollama_url=ollama_url,
        model_id=llm_model,
    )

    is_correct = verdict.get("is_correct", False)
    feedback = verdict.get("feedback", "")
    verification_explanation = verdict.get("explanation", "")

    # Step 6: Synthesize and play feedback
    print("\n[6/6] Text-to-Speech Feedback")
    print("-" * 60)
    feedback_text = f"{'Correct! ' if is_correct else 'Incorrect. '}{feedback} {explanation}"
    feedback_audio = await synthesize_speech(feedback_text)
    if play_response:
        play_audio(feedback_audio)

    # Print results
    print("\n" + "=" * 60)
    print("ROUND RESULTS")
    print("=" * 60)
    print(f"Question: {question}")
    print(f"Your answer: {user_answer}")
    print(f"Correct answer: {correct_answer}")
    print(f"Result: {'✓ CORRECT!' if is_correct else '✗ INCORRECT'}")
    print(f"Feedback: {feedback}")

    return {
        "round": round_number,
        "question": question,
        "user_answer": user_answer,
        "correct_answer": correct_answer,
        "is_correct": is_correct,
        "feedback": feedback,
        "explanation": explanation,
        "skip": False,
    }


async def interactive_trivia_quiz(
    duration: int = 5,
    sample_rate: int = 16000,
    category: str = "general knowledge",
    difficulty: str = "medium",
    num_rounds: int = 5,
    ollama_url: str = "http://localhost:11434",
    whisper_model: str = "base",
    llm_model: str = "granite4.1:3b",
    play_response: bool = True,
) -> None:
    """Run an interactive trivia quiz with multiple rounds.

    Args:
        duration: Recording duration per round in seconds
        sample_rate: Sample rate in Hz
        category: Trivia category
        difficulty: Question difficulty
        num_rounds: Number of trivia rounds
        ollama_url: Ollama server URL
        whisper_model: Whisper model size
        llm_model: LLM model name
        play_response: Whether to play audio responses
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE TRIVIA QUIZ")
    print("=" * 60)
    print(f"Category: {category}")
    print(f"Difficulty: {difficulty}")
    print(f"Rounds: {num_rounds}")

    results = []
    correct_count = 0

    for round_num in range(1, num_rounds + 1):
        try:
            result = await run_trivia_round(
                duration=duration,
                sample_rate=sample_rate,
                category=category,
                difficulty=difficulty,
                ollama_url=ollama_url,
                whisper_model=whisper_model,
                llm_model=llm_model,
                play_response=play_response,
                round_number=round_num,
            )

            if not result.get("skip"):
                results.append(result)
                if result.get("is_correct"):
                    correct_count += 1

            # Ask if user wants to continue
            if round_num < num_rounds:
                print(f"\n[Progress: {round_num}/{num_rounds}]")
                input("Press Enter to continue to the next question...")

        except KeyboardInterrupt:
            print("\n\nQuiz interrupted.")
            break
        except Exception as e:
            print(f"\nError in round {round_num}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    print("\n" + "=" * 60)
    print("QUIZ COMPLETE!")
    print("=" * 60)
    print(f"Score: {correct_count}/{len(results)} ({100 * correct_count // len(results) if results else 0}%)")
    print("\nSummary:")
    for r in results:
        status = "✓" if r["is_correct"] else "✗"
        print(f"  {status} Q: {r['question'][:50]}...")
        print(f"     Your answer: {r['user_answer']}")
        print(f"     Correct: {r['correct_answer']}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive trivia quiz with audio: generate questions, record answers, verify responses"
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
        "-c",
        "--category",
        default="general knowledge",
        help="Trivia category (general knowledge, history, science, geography, etc.)",
    )
    parser.add_argument(
        "--difficulty",
        default="medium",
        choices=["easy", "medium", "hard"],
        help="Question difficulty level (default: medium)",
    )
    parser.add_argument(
        "-n",
        "--num-rounds",
        type=int,
        default=5,
        help="Number of trivia rounds (default: 5)",
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
        "--no-play",
        action="store_true",
        help="Don't play audio responses",
    )
    parser.add_argument(
        "--voice",
        default="en-US-AriaNeural",
        help="Voice to use (default: en-US-AriaNeural)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive quiz mode",
    )

    args = parser.parse_args()

    try:
        if args.interactive:
            asyncio.run(
                interactive_trivia_quiz(
                    duration=args.duration,
                    sample_rate=args.sample_rate,
                    category=args.category,
                    difficulty=args.difficulty,
                    num_rounds=args.num_rounds,
                    ollama_url=args.ollama_url,
                    whisper_model=args.whisper_model,
                    llm_model=args.llm_model,
                    play_response=not args.no_play,
                )
            )
        else:
            # Single round mode
            asyncio.run(
                run_trivia_round(
                    duration=args.duration,
                    sample_rate=args.sample_rate,
                    category=args.category,
                    difficulty=args.difficulty,
                    ollama_url=args.ollama_url,
                    whisper_model=args.whisper_model,
                    llm_model=args.llm_model,
                    play_response=not args.no_play,
                    round_number=1,
                )
            )

    except Exception as e:
        print(f"\nTrivia quiz failed: {e}")
        print("\nMake sure:")
        print("  1. Ollama is running: ollama serve")
        print("  2. LLM model is pulled: ollama pull granite4.1:3b")
        print("  3. Dependencies installed:")
        print("     pip install openai-whisper edge-tts")
        sys.exit(1)


if __name__ == "__main__":
    main()
