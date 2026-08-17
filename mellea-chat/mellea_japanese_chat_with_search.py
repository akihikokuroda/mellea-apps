#!/usr/bin/env python3
"""Japanese audio chat with web search: Record → STT (Japanese) → LLM (with search) → TTS (Japanese)."""

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
    language: str = "ja",
) -> str:
    """Transcribe audio using OpenAI's Whisper model (local).

    Args:
        audio_bytes: WAV audio as bytes
        whisper_model: Whisper model size (tiny, base, small, medium, large)
        language: Language code (e.g., "ja" for Japanese)

    Returns:
        Transcribed text
    """
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
            print(f"警告: 音声がほぼ無音です (RMS: {audio_rms:.6f})")
            print("ヒント: より大きく、マイクに近い位置で話してください")

        result = model.transcribe(audio_array, language=language, fp16=False)
        text = result.get("text", "").strip()
        if not text:
            print("警告: Whisper からの文字起こしが空です")
            print("ヒント: 音声が不明確か、音量が小さい可能性があります")
        print(f"文字起こし: {text}")
        return text
    except Exception as e:
        print(f"音声文字起こしエラー: {e}")
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
        print("検索をスキップ: クエリが短すぎるか空です")
        return []

    query = query.strip()
    print(f"DuckDuckGo で検索しています: '{query}'...")
    try:
        # Run search in thread to avoid blocking
        import concurrent.futures
        import threading

        results = []

        def perform_search():
            try:
                with DDGS(timeout=10) as ddgs:
                    search_results = list(ddgs.text(query, max_results=max_results + 2))
                    return search_results
            except Exception as e:
                print(f"  検索中にエラーが発生しました: {e}")
                return []

        # Run in thread pool
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            search_results = await loop.run_in_executor(executor, perform_search)

        # Debug: Print raw results
        if debug:
            print(f"\n  [DEBUG] 生の検索結果数: {len(search_results)}")
            if search_results:
                print(f"  [DEBUG] 最初の結果のキー: {list(search_results[0].keys()) if search_results else 'N/A'}")
                print(f"  [DEBUG] 完全な生の結果:")
                for i, result in enumerate(search_results[:max_results], 1):
                    print(f"    結果 {i}:")
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
                    print(f"  [DEBUG] 結果 - has_title: {has_title}, has_body: {has_body}, "
                          f"title: '{result.get('title', '')[:30]}...', "
                          f"body: '{result.get('body', '')[:30]}...'")

                if has_title and has_body:
                    results.append({
                        "title": result.get("title", "")[:150],
                        "snippet": result.get("body", "")[:snippet_length],
                        "link": result.get("href", ""),
                    })

        if results:
            print(f"  → {len(results)} 件の検索結果が見つかりました")
            for i, r in enumerate(results[:max_results], 1):
                print(f"     {i}. {r['title'][:50]}...")
        else:
            print(f"  → 検索結果が見つかりません")
            if search_results:
                print(f"  → (注意: {len(search_results)} 件の結果を取得しましたが、タイトルとボディの両方を持つものはありません)")

        return results[:max_results]

    except Exception as e:
        print(f"警告: DuckDuckGo 検索エラー: {e}")
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

    context = "\n\n[ウェブ検索からのコンテキスト]\n"
    for i, result in enumerate(search_results, 1):
        title = result.get('title', '').strip()
        snippet = result.get('snippet', '').strip()
        link = result.get('link', '').strip()

        if title or snippet:
            context += f"\n{i}. {title}\n"
            if snippet:
                context += f"   {snippet}\n"
            if link:
                context += f"   出典: {link}"

    context += "\n[コンテキスト終了]\n"
    return context


async def send_to_mellea_llm_with_search(
    user_text: str,
    ollama_url: str = "http://localhost:11434",
    model_id: str = "granite4.1:3b",
    system_prompt: str = "You are a helpful assistant. Respond in Japanese.",
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
            print(f"{len(search_results)} 件の検索結果が見つかりました")

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

    print(f"Mellea LLM に送信しています ({ollama_url})...")
    print(f"モデル: {model_id}")
    print(f"ユーザー入力: {user_text}")

    # Debug: Show what's being sent to LLM
    if search_context:
        print(f"\n[DEBUG] LLM に送信される検索コンテキスト:")
        print(f"{search_context}")
    else:
        print(f"\n[DEBUG] 検索コンテキストなし (検索が無効またはコンテキストなし)")

    try:
        # Create action block with user text and search context
        action_text = user_text + search_context
        print(f"\n{'='*60}")
        print(f"[DEBUG] LLM に送信される完全なプロンプト")
        print(f"{'='*60}")
        print(f"結合されたプロンプトの長さ: {len(action_text)} 文字")
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
        print(f"アシスタント: {response}")
        return response
    except Exception as e:
        print(f"Mellea LLM エラー: {e}")
        import traceback
        traceback.print_exc()
        print("注意: Ollama が実行中で、モデルが取得されていることを確認してください:")
        print(f"  ollama serve")
        print(f"  ollama pull {model_id}")
        raise


async def synthesize_speech(
    text: str,
    output_file: str | None = None,
    rate: str = "+0%",
    voice: str = "ja-JP-NanamiNeural",
) -> bytes:
    """Synthesize text to speech using Edge TTS (Japanese).

    Args:
        text: Text to synthesize
        output_file: Optional output file path
        rate: Speech rate (e.g., "+0%", "+10%", "-10%")
        voice: Voice to use (default: ja-JP-NanamiNeural)

    Returns:
        Audio bytes in MP3 format
    """
    import io

    print("音声合成中...")
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
            print(f"音声を保存しました: {output_file}")

        return audio_bytes
    except Exception as e:
        print(f"音声合成エラー: {e}")
        raise


def play_audio(audio_bytes: bytes):
    """Play MP3 audio bytes through speakers.

    Args:
        audio_bytes: MP3 audio data as bytes
    """
    import io
    import subprocess
    import tempfile

    print("音声を再生しています...")
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

        print("音声再生が完了しました")

        # Clean up
        Path(tmp_path).unlink()
    except Exception as e:
        print(f"音声再生エラー: {e}")


async def process_single_turn(
    audio_bytes: bytes,
    ollama_url: str = "http://localhost:11434",
    whisper_model: str = "base",
    llm_model: str = "granite4.1:3b",
    system_prompt: str = "You are a helpful assistant. Respond in Japanese.",
    play_response: bool = True,
    turn_number: int = 1,
    japanese_voice: str = "ja-JP-NanamiNeural",
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
        japanese_voice: Japanese voice for TTS
        enable_search: Whether to enable DuckDuckGo search
        snippet_length: Maximum snippet length in characters

    Returns:
        Dict with transcription and response
    """
    print("\n" + "=" * 60)
    print(f"ターン {turn_number}")
    print("=" * 60)

    # Step 1: STT
    print("\n[1/4] 音声文字起こし (Whisper)")
    print("-" * 60)
    transcription = transcribe_with_whisper(audio_bytes, whisper_model, language="ja")

    # Check if transcription is empty
    if not transcription or len(transcription.strip()) < 2:
        print("エラー: 音声を文字起こしできませんでした。もう一度試してください。")
        return {"transcription": transcription, "response": None, "exit": False}

    # Check for exit commands (Japanese and English)
    user_input_lower = transcription.lower().strip()
    exit_words = ["bye", "quit", "exit", "goodbye", "さようなら", "バイ", "やめる", "終了", "終わり"]
    if any(word in user_input_lower for word in exit_words):
        return {"transcription": transcription, "response": None, "exit": True}

    # Step 2: LLM with Search
    print("\n[2/4] 言語モデル生成 (ウェブ検索あり)")
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
    print("\n[3/4] 音声合成 (Edge TTS)")
    print("-" * 60)
    response_audio_bytes = await synthesize_speech(response, voice=japanese_voice)

    # Step 4: Play response
    print("\n[4/4] 音声再生")
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
    print("ターン結果")
    print("=" * 60)
    print(f"ユーザー: {result['transcription']}")
    print(f"アシスタント: {result['response']}")

    return result


async def interactive_japanese_chat_with_search(
    duration: int = 5,
    sample_rate: int = 16000,
    ollama_url: str = "http://localhost:11434",
    whisper_model: str = "base",
    llm_model: str = "granite4.1:3b",
    system_prompt: str = "You are a helpful assistant. Respond in Japanese.",
    play_response: bool = True,
    japanese_voice: str = "ja-JP-NanamiNeural",
    enable_search: bool = True,
    snippet_length: int = 2000,
) -> None:
    """Interactive Japanese audio chat loop with web search.

    Records audio, transcribes in Japanese, searches the web, sends to LLM,
    synthesizes Japanese response, and plays it.
    Repeats until user says "さようなら", "終了", "bye", "quit", etc.

    Args:
        duration: Recording duration in seconds per turn
        sample_rate: Sample rate in Hz
        ollama_url: Ollama server URL
        whisper_model: Whisper model size
        llm_model: LLM model name
        system_prompt: System prompt for LLM
        play_response: Whether to play audio responses
        japanese_voice: Japanese voice for TTS
        enable_search: Whether to enable web search
        snippet_length: Maximum snippet length in characters
    """
    print("\n" + "=" * 60)
    print("日本語音声チャット (ウェブ検索機能付き)")
    print("=" * 60)
    print(f"ウェブ検索: {'有効' if enable_search else '無効'}")
    print(f"'さようなら'、'終了'、'bye'、'quit' などと言うと会話を終了できます")

    turn = 1
    while True:
        try:
            # Record audio
            print(f"\n[ターン {turn}] 録音中...")
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
                japanese_voice=japanese_voice,
                enable_search=enable_search,
                snippet_length=snippet_length,
            )

            # Check if user wants to exit
            if result.get("exit"):
                print("\n" + "=" * 60)
                print("さようなら!")
                print("=" * 60)
                break

            turn += 1

        except KeyboardInterrupt:
            print("\n\n会話が中断されました。")
            break
        except Exception as e:
            print(f"\nターン {turn} でエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Japanese audio chat with web search: STT (Japanese) → LLM → TTS (Japanese)"
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
        default="You are a helpful assistant. Respond in Japanese.",
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
        default="ja-JP-NanamiNeural",
        help="Japanese voice to use (default: ja-JP-NanamiNeural)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive chat mode (repeat until 'さようなら')",
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

    print("\n" + "=" * 60)
    print("日本語音声チャット (ウェブ検索機能付き) - 初期化中...")
    print("=" * 60)

    try:
        # Interactive mode
        if args.interactive:
            asyncio.run(
                interactive_japanese_chat_with_search(
                    duration=args.duration,
                    sample_rate=args.sample_rate,
                    ollama_url=args.ollama_url,
                    whisper_model=args.whisper_model,
                    llm_model=args.llm_model,
                    system_prompt=args.prompt,
                    play_response=not args.no_play,
                    japanese_voice=args.voice,
                    enable_search=not args.no_search,
                    snippet_length=args.snippet_length,
                )
            )
        else:
            # Single turn mode
            # Get input audio data
            if args.audio_file:
                print(f"音声ファイルを読み込んでいます: {args.audio_file}...")
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
                print(f"入力音声を保存しました: {args.save_input_audio}")

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
                    japanese_voice=args.voice,
                    enable_search=not args.no_search,
                    snippet_length=args.snippet_length,
                )
            )

    except Exception as e:
        print(f"\nパイプラインが失敗しました: {e}")
        print("\n確認してください:")
        print("  1. Ollama が実行中: ollama serve")
        print("  2. LLM モデルが取得済み: ollama pull granite4.1:3b")
        print("  3. 依存パッケージがインストール済み:")
        print("     pip install openai-whisper edge-tts ddgs")
        sys.exit(1)


if __name__ == "__main__":
    main()
