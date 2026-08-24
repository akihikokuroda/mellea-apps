#!/usr/bin/env python3
"""Japanese audio trivia quiz: Generate trivia question → Record answer → STT (Japanese) → LLM verification → TTS (Japanese)."""

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
    print(f"{duration}秒間録音します (早く停止するには Ctrl+C を押してください)...")
    try:
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
        )
        sd.wait()
        print(f"録音完了: {len(recording)} フレーム")
        return recording
    except KeyboardInterrupt:
        sd.stop()
        print("\n録音が早期に停止されました")
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
    """Transcribe audio using OpenAI's Whisper model (local, Japanese)."""
    import io
    import wave

    print(f"Whisper モデルを読み込んでいます: {whisper_model}...")
    try:
        model = whisper.load_model(whisper_model)
    except Exception as e:
        print(f"Whisper モデル読み込みエラー: {e}")
        raise

    print("音声を文字起こししています...")
    try:
        wav_file = io.BytesIO(audio_bytes)
        with wave.open(wav_file, 'rb') as wf:
            sample_rate = wf.getframerate()
            num_frames = wf.getnframes()
            audio_data = wf.readframes(num_frames)

        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        audio_rms = np.sqrt(np.mean(audio_array ** 2))
        if audio_rms < 0.01:
            print(f"警告: 音声が無音または非常に小さいようです (RMS: {audio_rms:.6f})")
            print("ヒント: マイクに近づいて、はっきりと話してください")

        result = model.transcribe(audio_array, language="ja", fp16=False)
        text = result.get("text", "").strip()
        if not text:
            print("警告: Whisper からの文字起こしが空です")
            print("ヒント: 音声が小さすぎるか不鮮明の可能性があります")
        print(f"文字起こし: {text}")
        return text
    except Exception as e:
        print(f"音声文字起こしエラー: {e}")
        raise


async def generate_trivia_question(
    category: str = "一般知識",
    difficulty: str = "medium",
    ollama_url: str = "http://localhost:11434",
    model_id: str = "granite4.1:3b",
    round_number: int = 1,
    previous_questions: list | None = None,
) -> dict:
    """Generate a Japanese trivia question using Mellea LLM.

    Args:
        category: Trivia category (一般知識, 歴史, 科学, etc.)
        difficulty: Difficulty level (easy, medium, hard)
        ollama_url: Ollama server URL
        model_id: Model name in Ollama
        round_number: Round number to help with variety
        previous_questions: List of previously asked questions to avoid repetition

    Returns:
        Dict with question, correct_answer, and explanation
    """
    backend = OllamaBackend(
        model_id=model_id,
        base_url=ollama_url,
    )

    ctx = SimpleContext()

    system_prompt = """You are a Japanese trivia question generator. Generate ONE completely unique and fresh trivia question in Japanese.
Make sure the question is different and not repetitive.
Output ONLY valid JSON, nothing else. Use this exact format:
{
    "question": "The question text in Japanese",
    "correct_answer": "The correct answer in Japanese",
    "alternatives": ["wrong answer 1 in Japanese", "wrong answer 2 in Japanese"],
    "explanation": "Why this is correct (in Japanese)"
}"""

    model_options = {
        ModelOption.TEMPERATURE: 0.95,
        ModelOption.SYSTEM_PROMPT: system_prompt,
    }

    variety_hint = f" 各問題は完全に異なる内容にしてください。これは問題 #{round_number} です。"

    previous_context = ""
    if previous_questions:
        prev_list = "\n".join([f"- {q}" for q in previous_questions[-3:]])  # Show last 3
        previous_context = f"\n\n前に聞いた問題:\n{prev_list}\n\nこれらの問題と異なる内容の問題を生成してください。"

    prompt = f"日本語で{difficulty}レベルの{category}のトリビア問題を生成してください。{variety_hint}前に聞いた問題と絶対に重複しないように、新しい話題を選んでください。{previous_context}"

    print(f"トリビア問題を生成中 ({difficulty} - {category})...")
    print(f"Mellea LLM に送信中 via Ollama ({ollama_url})...")
    print(f"モデル: {model_id}")

    try:
        action = CBlock(prompt)

        mot, gen_ctx = await mfuncs.aact(
            action, ctx, backend, strategy=None, model_options=model_options
        )

        response = await mot.avalue()
        print(f"生のレスポンス長: {len(response)} 文字")

        response_text = response.strip()

        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            print(f"抽出された JSON: {json_str[:100]}...")
            trivia_data = json.loads(json_str)

            if all(key in trivia_data for key in ["question", "correct_answer", "explanation"]):
                print(f"✓ トリビア問題を生成しました: {trivia_data['question'][:50]}...")
                return trivia_data

        print(f"警告: レスポンスから有効な JSON を抽出できませんでした")
        print(f"完全なレスポンス: {response}")
        raise ValueError("無効な JSON レスポンス")

    except Exception as e:
        print(f"トリビア問題生成エラー: {e}")
        print(f"より簡単なプロンプトで再試行中...")

        try:
            simple_prompt = f"日本語で{category}の{difficulty}トリビア問題を JSON 形式のみで生成してください。"
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
                    print(f"✓ トリビア問題を生成しました (再試行): {trivia_data['question'][:50]}...")
                    return trivia_data
        except:
            pass

        questions_bank = [
            {
                "question": "7 + 8 は何ですか?",
                "correct_answer": "15",
                "alternatives": ["14", "16"],
                "explanation": "7 + 8 = 15 です。"
            },
            {
                "question": "12 × 5 は何ですか?",
                "correct_answer": "60",
                "alternatives": ["50", "70"],
                "explanation": "12 × 5 = 60 です。"
            },
            {
                "question": "100 ÷ 4 は何ですか?",
                "correct_answer": "25",
                "alternatives": ["20", "30"],
                "explanation": "100 ÷ 4 = 25 です。"
            },
            {
                "question": "9 の二乗は何ですか?",
                "correct_answer": "81",
                "alternatives": ["72", "90"],
                "explanation": "9 の二乗 (9 × 9) = 81 です。"
            },
            {
                "question": "15 - 7 は何ですか?",
                "correct_answer": "8",
                "alternatives": ["7", "9"],
                "explanation": "15 - 7 = 8 です。"
            },
        ]

        selected = questions_bank[(round_number - 1) % len(questions_bank)]
        print(f"⚠ フォールバック問題を使用します: {selected['question']}")
        return selected


async def verify_answer(
    user_answer: str,
    correct_answer: str,
    question: str,
    explanation: str,
    ollama_url: str = "http://localhost:11434",
    model_id: str = "granite4.1:3b",
) -> dict:
    """Verify user answer using Mellea LLM (Japanese).

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

    system_prompt = """You are a strict Japanese trivia quiz judge. Determine if the user's answer is correct.
The answer must match the correct answer in meaning or be a very close variation.
Return a JSON response in Japanese with these exact fields:
{
    "is_correct": true or false,
    "explanation": "Why this answer is correct or incorrect (in Japanese)",
    "feedback": "Encouraging or constructive feedback for the user (in Japanese)"
}

Be strict about correctness - only accept answers that match the correct answer or are obviously equivalent. Return ONLY JSON, no other text."""

    model_options = {
        ModelOption.TEMPERATURE: 0.5,
        ModelOption.SYSTEM_PROMPT: system_prompt,
    }

    prompt = f"""質問: {question}

正解: {correct_answer}

ユーザーの回答: {user_answer}

正解の説明: {explanation}

指示:
- ユーザーの回答が正解と一致するか、または明らかに同等か確認してください
- スペルやわずかなバリエーションは許容しますが、本質的に異なる場合は不正解です
- 数値の場合、正確な数値が必要です
- 必ず JSON のみで応答してください"""

    print(f"回答を検証中...")
    print(f"ユーザーが言ったこと: '{user_answer}'")
    print(f"正解: '{correct_answer}'")

    try:
        action = CBlock(prompt)

        mot, gen_ctx = await mfuncs.aact(
            action, ctx, backend, strategy=None, model_options=model_options
        )

        response = await mot.avalue()

        try:
            verdict = json.loads(response)
            print(f"✓ 回答検証: {'正解' if verdict.get('is_correct') else '不正解'}")
            return verdict
        except json.JSONDecodeError:
            print(f"警告: 検証 JSON を解析できませんでした")
            print(f"レスポンス: {response}")
            is_correct = user_answer.lower().strip() == correct_answer.lower().strip()
            return {
                "is_correct": is_correct,
                "explanation": explanation,
                "feedback": "良い試みです!" if is_correct else "次回頑張ってください!"
            }

    except Exception as e:
        print(f"回答検証エラー: {e}")
        import traceback
        traceback.print_exc()
        raise


async def synthesize_speech(
    text: str,
    output_file: str | None = None,
    rate: str = "+0%",
    voice: str = "ja-JP-NanamiNeural",
) -> bytes:
    """Synthesize text to speech using Edge TTS (Japanese)."""
    import io

    print("音声合成中...")
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
            print(f"音声を保存しました: {output_file}")

        return audio_bytes
    except Exception as e:
        print(f"音声合成エラー: {e}")
        raise


def play_audio(audio_bytes: bytes):
    """Play MP3 audio bytes through speakers."""
    import io
    import subprocess
    import tempfile

    print("音声を再生しています...")
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        if sys.platform == "darwin":  # macOS
            subprocess.run(["afplay", tmp_path], check=True)
        elif sys.platform == "linux":
            subprocess.run(["aplay", tmp_path], check=True)
        elif sys.platform == "win32":
            import winsound
            winsound.PlaySound(tmp_path, winsound.SND_FILENAME)

        print("音声再生が完了しました")

        Path(tmp_path).unlink()
    except Exception as e:
        print(f"音声再生エラー: {e}")


async def run_trivia_round(
    duration: int = 5,
    sample_rate: int = 16000,
    category: str = "一般知識",
    difficulty: str = "medium",
    ollama_url: str = "http://localhost:11434",
    whisper_model: str = "base",
    llm_model: str = "granite4.1:3b",
    play_response: bool = True,
    round_number: int = 1,
    previous_questions: list | None = None,
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
        previous_questions: List of previously asked questions to avoid repetition

    Returns:
        Dict with question, user_answer, is_correct, and other metadata
    """
    print("\n" + "=" * 60)
    print(f"トリビア ラウンド {round_number}")
    print("=" * 60)

    print("\n[1/4] トリビア問題を生成中")
    print("-" * 60)
    trivia_data = await generate_trivia_question(
        category=category,
        difficulty=difficulty,
        ollama_url=ollama_url,
        model_id=llm_model,
        round_number=round_number,
        previous_questions=previous_questions,
    )

    question = trivia_data.get("question", "")
    correct_answer = trivia_data.get("correct_answer", "")
    explanation = trivia_data.get("explanation", "")

    print(f"Q: {question}")
    print(f"   (ヒント: {category} について考えてください)")

    print("\n[2/4] 音声合成で問題を読み上げ中")
    print("-" * 60)
    question_audio = await synthesize_speech(f"問題: {question}")
    if play_response:
        play_audio(question_audio)

    print("\n[3/4] 回答を録音中")
    print("-" * 60)
    print("問題に答えてください...")
    audio_data = record_audio(duration=duration, sample_rate=sample_rate, channels=1)
    audio_bytes = audio_to_wav_bytes(audio_data, sample_rate)

    print("\n[4/4] 回答を文字起こし中")
    print("-" * 60)
    user_answer = transcribe_with_whisper(audio_bytes, whisper_model)

    if not user_answer or len(user_answer.strip()) < 1:
        print("エラー: 回答を文字起こしできませんでした。はっきり話してもう一度試してください。")
        return {
            "question": question,
            "user_answer": None,
            "is_correct": False,
            "verification": None,
            "skip": True,
        }

    print("\n[5/5] 回答を検証中")
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

    print("\n[6/6] フィードバック音声合成中")
    print("-" * 60)
    feedback_text = f"{'正解です! ' if is_correct else '不正解です。 '}{feedback} {explanation}"
    feedback_audio = await synthesize_speech(feedback_text)
    if play_response:
        play_audio(feedback_audio)

    print("\n" + "=" * 60)
    print("ラウンド結果")
    print("=" * 60)
    print(f"質問: {question}")
    print(f"あなたの回答: {user_answer}")
    print(f"正解: {correct_answer}")
    print(f"結果: {'✓ 正解!' if is_correct else '✗ 不正解'}")
    print(f"フィードバック: {feedback}")

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
    category: str = "一般知識",
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
    print("日本語トリビアクイズ")
    print("=" * 60)
    print(f"カテゴリ: {category}")
    print(f"難易度: {difficulty}")
    print(f"ラウンド数: {num_rounds}")

    results = []
    correct_count = 0
    previous_questions = []

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
                previous_questions=previous_questions,
            )

            if not result.get("skip"):
                results.append(result)
                previous_questions.append(result["question"])
                if result.get("is_correct"):
                    correct_count += 1

            if round_num < num_rounds:
                print(f"\n[進捗: {round_num}/{num_rounds}]")
                input("次の問題に進むには Enter キーを押してください...")

        except KeyboardInterrupt:
            print("\n\nクイズが中断されました。")
            break
        except Exception as e:
            print(f"\nラウンド {round_num} でエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    print("クイズ完了!")
    print("=" * 60)
    print(f"スコア: {correct_count}/{len(results)} ({100 * correct_count // len(results) if results else 0}%)")
    print("\n概要:")
    for r in results:
        status = "✓" if r["is_correct"] else "✗"
        print(f"  {status} Q: {r['question'][:50]}...")
        print(f"     あなたの回答: {r['user_answer']}")
        print(f"     正解: {r['correct_answer']}")


def main():
    parser = argparse.ArgumentParser(
        description="日本語音声トリビアクイズ: 問題生成、回答録音、応答検証"
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
        default="一般知識",
        help="Trivia category (一般知識, 歴史, 科学, 地理, etc.)",
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
        default="ja-JP-NanamiNeural",
        help="Voice to use (default: ja-JP-NanamiNeural)",
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
        print(f"\n日本語トリビアクイズが失敗しました: {e}")
        print("\n確認してください:")
        print("  1. Ollama が実行中: ollama serve")
        print("  2. LLM モデルが取得済み: ollama pull granite4.1:3b")
        print("  3. 依存パッケージがインストール済み:")
        print("     pip install openai-whisper edge-tts")
        sys.exit(1)


if __name__ == "__main__":
    main()
