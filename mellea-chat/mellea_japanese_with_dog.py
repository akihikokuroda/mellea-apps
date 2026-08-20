#!/usr/bin/env python3
"""日本語音声パイプライン: 録音 → STT → LLM → TTS (犬の動き付き)"""

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
    print("エラー: 必要なライブラリがインストールされていません")
    print("インストール: pip install sounddevice soundfile numpy openai-whisper")
    sys.exit(1)

try:
    from pydub import AudioSegment
    from pydub.generators import Sine
except ImportError:
    print("エラー: pydubがインストールされていません")
    print("インストール: pip install pydub")
    sys.exit(1)

try:
    import edge_tts
    import asyncio as aio
except ImportError:
    print("エラー: edge-ttsがインストールされていません")
    print("インストール: pip install edge-tts")
    sys.exit(1)

try:
    from mellea.backends.ollama import OllamaModelBackend as OllamaBackend
    from mellea.backends.model_options import ModelOption
    from mellea.stdlib.components import CBlock
    from mellea.stdlib.context import SimpleContext
    import mellea.stdlib.functional as mfuncs
except ImportError:
    print("エラー: Melleaがインストールされていません")
    print("インストール: pip install mellea[backends]")
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

try:
    from animation_exporter import AnimationRecorder
except ImportError:
    AnimationRecorder = None


def record_audio(
    duration: int = 5,
    sample_rate: int = 16000,
    channels: int = 1,
) -> np.ndarray:
    """マイクから音声を録音する。

    Args:
        duration: 録音時間（秒）
        sample_rate: サンプリングレート（Hz）
        channels: チャンネル数

    Returns:
        音声データのNumPy配列
    """
    print(f"{duration}秒間の録音（早く終了するにはCtrl+Cを押してください）...")
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
        print("\n録音は早期に停止しました")
        return recording


def audio_to_wav_bytes(audio_data: np.ndarray, sample_rate: int = 16000) -> bytes:
    """音声配列をWAVバイトに変換する。

    Args:
        audio_data: 音声のNumPy配列
        sample_rate: サンプリングレート（Hz）

    Returns:
        WAVファイルのバイト
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
    """OpenAIのWhisperモデルを使用して音声を転記する（ローカル）。

    Args:
        audio_bytes: WAV音声バイト
        whisper_model: Whisperモデルサイズ (tiny, base, small, medium, large)

    Returns:
        転記されたテキスト
    """
    import io
    import wave

    print(f"Whisperモデルをロード中: {whisper_model}...")
    try:
        model = whisper.load_model(whisper_model)
    except Exception as e:
        print(f"エラー: Whisperモデルのロードに失敗しました: {e}")
        raise

    print("音声を転記中...")
    try:
        # WAVバイトから生の音声データを抽出
        wav_file = io.BytesIO(audio_bytes)
        with wave.open(wav_file, 'rb') as wf:
            sample_rate = wf.getframerate()
            num_frames = wf.getnframes()
            audio_data = wf.readframes(num_frames)

        # バイトをnumpy配列に変換
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        result = model.transcribe(audio_array, language="ja", fp16=False)
        text = result.get("text", "").strip()
        if not text:
            print("警告: Whisperからの転記が空です")
        print(f"転記: {text}")
        return text
    except Exception as e:
        print(f"エラー: 音声の転記に失敗しました: {e}")
        raise


async def send_to_mellea_llm(
    user_text: str,
    ollama_url: str = "http://localhost:11434",
    model_id: str = "granite4.1:3b",
    system_prompt: str = "あなたは親切なアシスタントです。",
    enable_dog_movements: bool = True,
) -> tuple[str, list[str]]:
    """転記されたテキストをMellea LLMに送信する（Ollama経由）。

    Args:
        user_text: ユーザーメッセージ
        ollama_url: OllamaサーバーのURL
        model_id: Ollamaのモデル名
        system_prompt: システムプロンプト
        enable_dog_movements: LLM応答で犬の動きコマンドを有効にするかどうか

    Returns:
        (生成された応答, 動きコマンドのリスト)のタプル
    """
    # Ollama用のMelleaバックエンドを作成
    backend = OllamaBackend(
        model_id=model_id,
        base_url=ollama_url,
    )

    # 非同期用のシンプルコンテキストを作成
    ctx = SimpleContext()

    # 犬の動きのシステムプロンプトを拡張
    enhanced_prompt = system_prompt
    if enable_dog_movements:
        enhanced_prompt += (
            "\n\n犬のおもちゃを動かしたい場合は、応答に[MOVE: command]形式の動きコマンドを含めてください。"
            "常に動きコマンドの前後に会話テキストを含めてください。動きコマンドだけの応答はしないでください。\n"
            "利用可能なコマンド:\n"
            "- tail_wag:style=fast_loose|slow_wide,duration=<ms>,swing_arc=<degrees>\n"
            "- head_tilt:angle=<degrees>,duration=<ms>\n"
            "- walk:gait=steady_walk|fast_trot,strides=<count>,stride_length=<mm>\n"
            "- beg:hip_angle=<degrees>,paw_lift=<mm>,hold_time=<ms>\n"
            "- ear_twitch:type=vertical|flaring,count=<cycles>,frequency=<hz>\n"
            "- pant:jaw_travel=<mm>,cycles=<count>,cycle_duration=<ms>\n"
            "- begging:rear_lift=<mm>,paw_lift=<mm>,hold_time=<ms>\n"
            "- sit:spine_angle=<degrees>,head_nod=<degrees>,sit_time=<ms>\n"
            "例: 元気です！ [MOVE: tail_wag:style=fast_loose,duration=1000]"
        )

    # 拡張されたシステムプロンプトのモデルオプション
    model_options = {
        ModelOption.TEMPERATURE: 0.7,
        ModelOption.SYSTEM_PROMPT: enhanced_prompt,
    }

    print(f"Mellea LLMに送信中（Ollama: {ollama_url}）...")
    print(f"モデル: {model_id}")
    print(f"ユーザー: {user_text}")

    try:
        # ユーザーテキストでアクションブロックを作成
        action = CBlock(user_text)

        # Mellealとollama バックエンドで応答を生成
        # aactは (ModelOutputThunk, Context) を返す
        mot, gen_ctx = await mfuncs.aact(
            action, ctx, backend, strategy=None, model_options=model_options
        )

        # thunkを解決して実際の文字列を取得
        raw_response = await mot.avalue()
        print(f"アシスタント（生）: {raw_response}")

        # 有効な場合は犬の動きを解析
        if enable_dog_movements:
            cleaned_response, movements = parse_llm_response_for_movements(raw_response)
            print(f"アシスタント: {cleaned_response}")
            if movements:
                print(f"犬の動き検出: {movements}")
            return cleaned_response, movements
        else:
            return raw_response, []

    except Exception as e:
        print(f"エラー: Mellea LLMの呼び出しに失敗しました: {e}")
        import traceback
        traceback.print_exc()
        print("注意: Ollamaが実行中で、モデルがプルされていることを確認してください:")
        print(f"  ollama serve")
        print(f"  ollama pull {model_id}")
        raise


async def synthesize_speech(
    text: str,
    output_file: str | None = None,
    rate: str = "+0%",
    voice: str = "ja-JP-NanamiNeural",
) -> bytes:
    """Edge TTSを使用してテキストを音声に合成する。

    Args:
        text: 合成するテキスト
        output_file: オプションの出力ファイルパス
        rate: 音声レート（例: "+0%", "+10%", "-10%"）
        voice: 使用する音声（デフォルト: ja-JP-NanakaNeural）

    Returns:
        MP3形式の音声バイト
    """
    import io
    import asyncio

    print("音声を合成中...")

    # 空のテキストを処理
    if not text or not text.strip():
        print("警告: TTS用に提供されたテキストが空です。空のバイトを返しています")
        return b""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)

            # 音声チャンクを収集
            audio_data = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.write(chunk["data"])

            audio_bytes = audio_data.getvalue()

            if not audio_bytes:
                if attempt < max_retries - 1:
                    print(f"警告: 空の音声データ。再試行 {attempt + 1}/{max_retries - 1}...")
                    await asyncio.sleep(1)
                    continue
                else:
                    print("エラー: Edge TTSが音声データを返しませんでした")
                    return b""

            if output_file:
                with open(output_file, 'wb') as f:
                    f.write(audio_bytes)
                print(f"音声を保存しました: {output_file}")

            return audio_bytes
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"エラー（再試行中）: {e}")
                await asyncio.sleep(1)
            else:
                print(f"エラー: 音声の合成に失敗しました: {e}")
                print("ヒント: インターネット接続を確認してください。Edge TTSサービスが利用可能か確認してください。")
                return b""

    return b""


def play_audio(audio_bytes: bytes):
    """MP3音声バイトをスピーカーで再生する。

    Args:
        audio_bytes: MP3音声データバイト
    """
    import io
    import subprocess
    import tempfile

    # 音声が空かどうかを確認
    if not audio_bytes or len(audio_bytes) == 0:
        print("警告: 再生する音声がありません（応答が空です）")
        return

    print("音声を再生中...")
    try:
        # 一時ファイルに保存して再生
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # システムオーディオプレイヤーを使用
        if sys.platform == "darwin":  # macOS
            subprocess.run(["afplay", tmp_path], check=True)
        elif sys.platform == "linux":
            subprocess.run(["aplay", tmp_path], check=True)
        elif sys.platform == "win32":
            import winsound
            winsound.PlaySound(tmp_path, winsound.SND_FILENAME)

        print("音声再生完了")

        # クリーンアップ
        Path(tmp_path).unlink()
    except Exception as e:
        print(f"エラー: 音声再生に失敗しました: {e}")


def execute_dog_movement(movement_command: str) -> list[MotionFrame]:
    """コマンド文字列に基づいて犬の動きを実行する。

    Args:
        movement_command: "motion_type:param1=value1,param2=value2"形式のコマンド
                         例: "tail_wag:style=fast_loose,duration=1000"

    Returns:
        動きのMotionFrameオブジェクトのリスト
    """
    print(f"犬の動き: {movement_command}")

    parts = movement_command.split(":")
    if len(parts) < 1:
        print("無効な動きコマンド形式")
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
            print(f"不明な動きタイプ: {motion_type}")
            return []

    except Exception as e:
        print(f"エラー: 犬の動きの実行に失敗しました: {e}")
        return []


def parse_llm_response_for_movements(response: str) -> tuple[str, list[str]]:
    """LLM応答から犬の動きコマンドを抽出する。

    [MOVE: command] または <<MOVE: command>> 形式の動きコマンドを探す

    Args:
        response: 生のLLM応答テキスト

    Returns:
        (クリーンされた応答, 動きコマンドのリスト)のタプル
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
    """LLM応答からすべての犬の動きを実行する。

    Args:
        movement_commands: 動きコマンド文字列のリスト

    Returns:
        実行ステータス、フレームカウント、および実際のフレームの辞書
    """
    execution_results = {}

    for cmd in movement_commands:
        frames = execute_dog_movement(cmd)
        execution_results[cmd] = {
            "frames": len(frames),
            "duration_ms": frames[-1].timestamp_ms if frames else 0,
            "frame_data": frames,
        }

    return execution_results


async def process_single_turn(
    audio_bytes: bytes,
    ollama_url: str = "http://localhost:11434",
    whisper_model: str = "base",
    llm_model: str = "granite4.1:3b",
    system_prompt: str = "あなたは親切なアシスタントです。",
    play_response: bool = True,
    turn_number: int = 1,
    enable_dog_movements: bool = True,
    animation_recorder = None,
) -> dict:
    """1つの会話ターンを処理する: STT → LLM → 犬の動き → TTS → 再生。

    Args:
        audio_bytes: 入力WAV音声バイト
        ollama_url: OllamaサーバーのURL
        whisper_model: Whisperモデルサイズ
        llm_model: LLMモデル名
        system_prompt: LLMのシステムプロンプト
        play_response: 音声応答を再生するかどうか
        turn_number: 表示用のターン番号
        animation_recorder: 動きをエクスポートするためのオプションのAnimationRecorder
        enable_dog_movements: LLMから犬の動きを有効にするかどうか

    Returns:
        転記、応答、および終了フラグを含む辞書
    """
    print("\n" + "=" * 60)
    print(f"ターン {turn_number}")
    print("=" * 60)

    # ステップ1: STT
    print("\n[1/5] 音声テキスト変換（Whisper）")
    print("-" * 60)
    transcription = transcribe_with_whisper(audio_bytes, whisper_model)

    # 転記が空かどうかを確認
    if not transcription or len(transcription.strip()) < 2:
        print("エラー: 音声を転記できませんでした。はっきりと話してからもう一度試してください。")
        return {"transcription": transcription, "response": None, "exit": False}

    # 終了コマンドを確認
    user_input_lower = transcription.lower().strip()
    if any(word in user_input_lower for word in ["さようなら", "終了", "バイ", "じゃあね"]):
        return {"transcription": transcription, "response": None, "exit": True}

    # ステップ2: LLM
    print("\n[2/5] 言語モデル生成（Mellea + Ollama）")
    print("-" * 60)
    response, movements = await send_to_mellea_llm(
        transcription,
        ollama_url=ollama_url,
        model_id=llm_model,
        system_prompt=system_prompt,
        enable_dog_movements=enable_dog_movements,
    )

    # 動き抽出後に応答が空かどうかを確認
    if not response or not response.strip():
        print("警告: LLMが動きコマンドのみで、テキスト応答がありません")
        response = "（犬は話さずに動きました）"

    # ステップ3: 犬の動きを実行
    movement_results = {}
    if enable_dog_movements and movements:
        print("\n[3/5] 犬の動き")
        print("-" * 60)
        movement_results = await execute_llm_movements(movements)
        for cmd, result in movement_results.items():
            print(f"  {cmd}: {result['frames']} フレーム, {result['duration_ms']}ms")

        # 動きが利用可能な場合は記録
        if animation_recorder:
            for cmd, result in movement_results.items():
                animation_recorder.record_movement(
                    movement_name=cmd.split(':')[0],
                    frames=result['frame_data'],
                    response_text=response,
                    turn_number=turn_number
                )

    # ステップ4: TTS
    print("\n[4/5] テキスト音声合成（Edge TTS）")
    print("-" * 60)
    response_audio_bytes = await synthesize_speech(response, voice="ja-JP-NanamiNeural")

    # ステップ5: 応答を再生
    print("\n[5/5] 音声再生")
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
    print("ターン結果")
    print("=" * 60)
    print(f"ユーザー: {result['transcription']}")
    print(f"アシスタント: {result['response']}")

    return result


async def interactive_audio_chat_with_dog(
    duration: int = 5,
    sample_rate: int = 16000,
    ollama_url: str = "http://localhost:11434",
    whisper_model: str = "base",
    llm_model: str = "granite4.1:3b",
    system_prompt: str = "あなたは親切なアシスタントです。",
    play_response: bool = True,
    enable_dog_movements: bool = True,
    animation_recorder = None,
) -> None:
    """犬の動きを持つインタラクティブ音声チャットループ。

    音声を録音し、転記し、LLMに送信し、犬の動きを実行し、
    応答を合成して再生します。ユーザーが「さようなら」などを言うまで繰り返します。

    Args:
        duration: ターンごとの録音時間（秒）
        sample_rate: サンプリングレート（Hz）
        ollama_url: OllamaサーバーのURL
        whisper_model: Whisperモデルサイズ
        animation_recorder: 動きをエクスポートするためのオプションのAnimationRecorder
        llm_model: LLMモデル名
        system_prompt: LLMのシステムプロンプト
        play_response: 音声応答を再生するかどうか
        enable_dog_movements: LLMから犬の動きを有効にするかどうか
    """
    print("\n" + "=" * 60)
    print("犬の動きを持つインタラクティブ音声チャット")
    print("=" * 60)
    print(f"犬の動き: {'有効' if enable_dog_movements else '無効'}")
    print(f"「さようなら」、「終了」、「バイ」、または「じゃあね」と言って会話を終了してください")

    turn = 1
    while True:
        try:
            # 音声を録音
            print(f"\n[ターン {turn}] 録音中...")
            audio_data = record_audio(
                duration=duration,
                sample_rate=sample_rate,
                channels=1,
            )
            audio_bytes = audio_to_wav_bytes(audio_data, sample_rate)

            # ターンを処理
            result = await process_single_turn(
                audio_bytes,
                ollama_url=ollama_url,
                whisper_model=whisper_model,
                llm_model=llm_model,
                system_prompt=system_prompt,
                play_response=play_response,
                turn_number=turn,
                enable_dog_movements=enable_dog_movements,
                animation_recorder=animation_recorder,
            )

            # ユーザーが終了したいかどうかを確認
            if result.get("exit"):
                print("\n" + "=" * 60)
                print("さようなら!")
                print("=" * 60)
                # レコーダーがアクティブな場合はアニメーションを保存
                if animation_recorder:
                    print("\nアニメーションを保存しています...")
                    saved_files = animation_recorder.save_all()
                    print(f"✓ {len(saved_files)}個のアニメーションファイルを保存しました")
                    for file_path in saved_files:
                        print(f"  - {file_path}")
                break

            turn += 1

        except KeyboardInterrupt:
            print("\n\n会話が中断しました。")
            break
        except Exception as e:
            print(f"\nターン {turn} でエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            continue


async def full_audio_response_pipeline(
    audio_bytes: bytes,
    ollama_url: str = "http://localhost:11434",
    whisper_model: str = "base",
    llm_model: str = "granite4.1:3b",
    system_prompt: str = "あなたは親切なアシスタントです。",
    output_audio_file: str | None = None,
    play_response: bool = True,
    enable_dog_movements: bool = True,
) -> dict:
    """完全なパイプライン: 音声 → STT → LLM → 犬の動き → TTS → 再生。

    Args:
        audio_bytes: 入力WAV音声バイト
        ollama_url: OllamaサーバーのURL
        whisper_model: Whisperモデルサイズ
        llm_model: LLMモデル名
        system_prompt: LLMのシステムプロンプト
        output_audio_file: 応答音声を保存するオプションのファイル
        play_response: 音声応答を再生するかどうか
        enable_dog_movements: LLMから犬の動きを有効にするかどうか

    Returns:
        転記、応答、音声情報、および動きを含む辞書
    """
    print("\n" + "=" * 60)
    print("完全音声応答パイプライン（犬の動き付き）")
    print("=" * 60)

    # ステップ1: STT
    print("\n[1/5] 音声テキスト変換（Whisper）")
    print("-" * 60)
    transcription = transcribe_with_whisper(audio_bytes, whisper_model)

    # ステップ2: LLM
    print("\n[2/5] 言語モデル生成（Mellea + Ollama）")
    print("-" * 60)
    response, movements = await send_to_mellea_llm(
        transcription,
        ollama_url=ollama_url,
        model_id=llm_model,
        system_prompt=system_prompt,
        enable_dog_movements=enable_dog_movements,
    )

    # 動き抽出後に応答が空かどうかを確認
    if not response or not response.strip():
        print("警告: LLMが動きコマンドのみで、テキスト応答がありません")
        response = "（犬は話さずに動きました）"

    # ステップ3: 犬の動きを実行
    movement_results = {}
    if enable_dog_movements and movements:
        print("\n[3/5] 犬の動き")
        print("-" * 60)
        movement_results = await execute_llm_movements(movements)
        for cmd, result in movement_results.items():
            print(f"  {cmd}: {result['frames']} フレーム, {result['duration_ms']}ms")

    # ステップ4: TTS
    print("\n[4/5] テキスト音声合成（Edge TTS）")
    print("-" * 60)
    response_audio_bytes = await synthesize_speech(response, output_file=output_audio_file, voice="ja-JP-NanakaNeural")

    # ステップ5: 応答を再生
    print("\n[5/5] 音声再生")
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
    print("結果")
    print("=" * 60)
    print(f"ユーザー入力: {result['transcription']}")
    print(f"LLM応答: {result['response']}")
    print(f"応答音声: {result['response_audio_bytes']} バイト")
    if movements:
        print(f"犬の動き: {len(movements)} 個のコマンドを実行しました")
    if output_audio_file:
        print(f"保存先: {result['output_file']}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="完全音声パイプライン: 録音 → STT → LLM → TTS → 再生（日本語、犬の動き付き）"
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=5,
        help="録音時間（秒）（デフォルト: 5）",
    )
    parser.add_argument(
        "-r",
        "--sample-rate",
        type=int,
        default=16000,
        help="サンプリングレート（Hz）（デフォルト: 16000）",
    )
    parser.add_argument(
        "-f",
        "--audio-file",
        help="録音の代わりに処理するWAVファイルのパス",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="OllamaサーバーのURL（デフォルト: http://localhost:11434）",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisperモデルサイズ: tiny, base, small, medium, large（デフォルト: base）",
    )
    parser.add_argument(
        "--llm-model",
        default="granite4.1:3b",
        help="Ollama LLMモデル名（デフォルト: granite4.1:3b）",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        default="あなたは親切で楽しいアシスタントです。日本語で応答してください。",
        help="LLMのシステムプロンプト",
    )
    parser.add_argument(
        "-s",
        "--save-input-audio",
        help="入力音声をファイルに保存",
    )
    parser.add_argument(
        "-o",
        "--output-audio",
        help="応答音声をファイルに保存",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="応答音声を再生しない",
    )
    parser.add_argument(
        "--speech-rate",
        default="+0%",
        help="音声レート: -50%% から +50%%（デフォルト: +0%%）",
    )
    parser.add_argument(
        "--voice",
        default="ja-JP-NanakaNeural",
        help="使用する音声（デフォルト: ja-JP-NanakaNeural）",
    )
    parser.add_argument(
        "--enable-dog-movements",
        action="store_true",
        default=True,
        help="LLMから犬の動きコマンドを有効にする（デフォルト: 有効）",
    )
    parser.add_argument(
        "--disable-dog-movements",
        action="store_true",
        help="LLMから犬の動きコマンドを無効にする",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="インタラクティブチャットモード（「さようなら」まで繰り返す）",
    )
    parser.add_argument(
        "--export-animations",
        action="store_true",
        help="犬の動きをJSONファイルとしてエクスポート（--interactive が必要）",
    )
    parser.add_argument(
        "--animations-dir",
        default="dog_animations",
        help="アニメーションJSONファイルを保存するディレクトリ（デフォルト: dog_animations）",
    )

    args = parser.parse_args()
    enable_movements = not args.disable_dog_movements

    try:
        # アニメーションレコーダーを作成（リクエストされた場合）
        animation_recorder = None
        if args.export_animations:
            if not AnimationRecorder:
                print("エラー: animation_exporter モジュールが利用できません")
                print("同じディレクトリに animation_exporter.py があることを確認してください")
                sys.exit(1)
            animation_recorder = AnimationRecorder(export_dir=args.animations_dir)
            print(f"✓ アニメーション記録を有効にしました（保存先: {args.animations_dir}）")

        # インタラクティブモード
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
                    animation_recorder=animation_recorder,
                )
            )
            return

        # シングルターンモード
        # 入力音声データを取得
        if args.audio_file:
            print(f"{args.audio_file}から音声を読み込み中...")
            audio_data, sample_rate = sf.read(args.audio_file, dtype="int16")
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]  # ステレオからモノに変換
            audio_bytes = audio_to_wav_bytes(audio_data, sample_rate)
        else:
            # マイクから録音
            audio_data = record_audio(
                duration=args.duration,
                sample_rate=args.sample_rate,
                channels=1,
            )
            audio_bytes = audio_to_wav_bytes(audio_data, args.sample_rate)

        # リクエストされた場合、入力音声を保存
        if args.save_input_audio:
            Path(args.save_input_audio).parent.mkdir(parents=True, exist_ok=True)
            sf.write(args.save_input_audio, audio_data, args.sample_rate)
            print(f"入力音声を保存しました: {args.save_input_audio}")

        # パイプラインで処理
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
        print(f"\nパイプラインが失敗しました: {e}")
        print("\n以下を確認してください:")
        print("  1. Ollamaが実行中: ollama serve")
        print("  2. LLMモデルがプルされている: ollama pull granite4.1:3b")
        print("  3. 依存関係がインストールされている:")
        print("     pip install openai-whisper edge-tts")
        sys.exit(1)


if __name__ == "__main__":
    main()
