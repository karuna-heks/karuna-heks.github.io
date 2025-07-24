#!/usr/bin/env python
# transcribe_zoom.py
"""
Обрабатывает либо папку с мульти-трек аудио (каждый файл = свой спикер),
либо одиночный файл с несколькими говорящими (диаризация).
Usage:
    python transcribe_zoom.py --input path/to/file_or_folder \
                              --model base \
                              --language ru \
                              --hf_token YOUR_HF_TOKEN
"""
import argparse
import json
import datetime
import collections
from pathlib import Path
import re
import whisperx
from faster_whisper import WhisperModel

FRAME_MS = 10                  # длительность одной PCM-рамки
TS_FMT = "%Y.%m.%d %H:%M:%S.%f"

# ---------- базовые функции ----------

def load_model(name="base", lang="ru", device="cuda", compute="auto"):
    print(f"→ loading Whisper {name} on {device} (compute_type={compute})")
    # return whisperx.load_model(
    #     name,
    #     device=device,
    #     language=lang,
    #     compute_type=compute,
    #     vad_method="silero", #"silero", None            # какой VAD‑модуль использовать
    #     vad_options={                   # всё, что должно попасть во «внутренний» transcribe
    #         # "vad_filter": True,         # включаем VAD‑фильтр
    #         # "vad_parameters": {         # тонкая настройка Silero‑VAD
    #         #     "threshold": 0.35, 
    #         #     "min_silence_duration_ms": 50,
    #         #     "speech_pad_ms": 0,
    #         #     "max_speech_duration_s":   10 # 6
    #         # },
    #         # "threshold": 0.35, 
    #         # "min_silence_duration_ms": 50,
    #         # "speech_pad_ms": 0,
    #         # "max_speech_duration_s":   6 # 6
    #         "vad_onset": 0.35,
    #         "chunk_size": 4,
    #     },
    #     asr_options={
    #         # "chunk_length": 6,
    #         "condition_on_previous_text": False,
    #         "word_timestamps": True
    #     }
    # )

    return WhisperModel(name, device=device, compute_type=compute)

def transcribe(model, audio_path):
    """ASR с тайм-кодами слов."""
    # для whisperx
    # return model.transcribe(audio_path, 
    #                         batch_size=16,
    #                         # vad_filter=True,
    #                         # vad_parameters=dict(
    #                         #     min_silence_duration_ms=250,
    #                         #     max_speech_duration_s=10
    #                         # )
    #                         )

    # для faster-whisper
    segments, info = model.transcribe(audio_path,
                            # batch_size=16,
                            vad_filter=True,
                            vad_parameters={
                                "min_silence_duration_ms": 200,
                                "max_speech_duration_s": 30,
                                "speech_pad_ms": 400,
                            },
                            word_timestamps=True,
                            condition_on_previous_text=True,
                            )
    seg_dicts = []
    for seg in segments:                       # Segment dataclass
        seg_dicts.append({
            "id":    seg.id,
            "start": seg.start,
            "end":   seg.end,
            "text":  seg.text,
            "words": [
                {
                    "start": w.start,
                    "end":   w.end,
                    "text":  w.word
                } for w in (seg.words or [])
            ],
        })
    return {"segments": seg_dicts, "language": info.language}


def find_log_gaps(ts_list, gap_ms=200):
    """
    ts_list: список datetime‑штампов (1 штамп = 1 аудиокадр FRAME_MS).
    Возвращает индексы, за которыми наступает пауза ≥ gap_ms.
    """
    gaps = []
    for i in range(1, len(ts_list)):
        if (ts_list[i] - ts_list[i-1]).total_seconds()*1000 >= gap_ms:
            gaps.append(i)            # пауза начинается ПЕРЕД кадром i
    return gaps


def split_segment_by_log(seg, ts_list, gaps_idx, node, frame_ms=10):
    """
    Разбиваем сегмент faster‑whisper по разрывам, найденным в логах.
    Возвращает list[dict] в формате faster‑whisper/WhisperX.
    """
    if not gaps_idx:                 # разрывов нет → вернуть как есть
        seg["speaker"] = node
        return [seg]

    gaps_idx = sorted(gaps_idx)      # на всякий случай
    gap_ptr  = 0                     # указатель на 'текущий' разрыв

    def make_segment(words_buf):
        return {
            "start": words_buf[0]["start"],
            "end":   words_buf[-1]["end"],
            "text":  "".join(w["text"] for w in words_buf),
            "words": words_buf,
            "speaker": node,
        }

    new_segs, buf = [], []
    for w in seg["words"]:
        frame_idx = round(w["start"] * 1000 / frame_ms)

        # прошли (или точно попали в) текущий разрыв?
        while gap_ptr < len(gaps_idx) and frame_idx >= gaps_idx[gap_ptr]:
            if buf:
                new_segs.append(make_segment(buf))
                buf = []
            gap_ptr += 1             # переключаемся на следующий разрыв

        buf.append(w)

    if buf:                          # финальный хвост
        new_segs.append(make_segment(buf))

    return new_segs


def abs_time(word_start_local: float, ts_list) -> datetime.datetime:
    frame_idx = round(word_start_local * 1000 / FRAME_MS)
    return ts_list[min(frame_idx, len(ts_list)-1)]


def diarize(audio_path, device="cuda", hf_token=None):
    pipe = whisperx.DiarizationPipeline(device=device, hf_token=hf_token)
    return pipe(audio_path)

def apply_diarization(asr_result, diarization_result):
    return whisperx.assign_word_speakers(diarization_result, asr_result)

def segments_to_dialogue(segments):
    """Собираем реплики вида 'Speaker X: текст'."""
    lines = []
    for s in segments:
        spk = s.get("speaker", "Speaker")
        abs_start = s.get("abs_start", "HH:MM:SS")
        lines.append(f"[{abs_start}] {spk}: {s['text'].strip()}")
    return "\n".join(lines)

# ---------- сценарии обработки ----------

def single_track(file_path, args):
    model = load_model(args.model, args.language, args.device, args.compute_type)
    asr = transcribe(model, file_path)
    diar = diarize(file_path, args.device, args.hf_token)
    merged = apply_diarization(asr, diar)
    txt = segments_to_dialogue(merged["segments"])
    out = Path(file_path).with_suffix(".dialogue.txt")
    out.write_text(txt, "utf-8")
    print(f"✓ dialogue saved to {out}")


def get_meeting_event_log(folder) -> list:
    by_node = collections.defaultdict(list)
    json_paths = list(Path(folder).glob("*.json"))
    assert len(json_paths) == 1
    with open(json_paths[0], "r") as f:
        meeting_event_log = json.loads(f.read())
        for rec in meeting_event_log:
            if rec["event"] != "on_one_way_audio_raw_data_received_callback":
                continue
            node = rec["node_id"]
            ts = datetime.datetime.strptime(rec["ts"], TS_FMT)
            by_node[node].append(ts)
            # print(f"added new node: {node} - {ts}")
    return by_node


def to_absolute(seg_start_local, ts_list):
    """seg_start_local -- float секунд от начала WAV."""
    frame_idx = round(seg_start_local * 1000 / FRAME_MS)
    try:
        return ts_list[frame_idx]
    except IndexError:
        # Whisper мог отбросить первые-несколько-тихих кадров,
        # поэтому fallback: последний валидный штамп + δ
        delta = (seg_start_local*1000 - frame_idx*FRAME_MS) / 1000
        return ts_list[-1] + datetime.timedelta(seconds=delta)


def id_from_wav(path: Path) -> str:
    """
    «user_16778240_20250710_141645.wav»  ->  '16778240'
    «track_5555.wav»                     ->  '5555'
    «123456.wav»                         ->  '123456'
    """
    m = re.search(r'(\d+)', path.stem)   # первая группа цифр
    return m.group(1) if m else path.stem


# def multi_track(folder, args):
#     model = load_model(args.model, args.language, args.device, args.compute_type)
#     ts_map = get_meeting_event_log(folder)

#     all_segments = []
#     for audio in sorted(Path(folder).glob("*.wav")):        # адаптируйте расширение
#         # node = audio.stem.split(".")[0]
#         node = id_from_wav(audio)

#         print("node_id в WAV  :", node)
#         print("есть в ts_map? :", node in ts_map)
#         if node in ts_map:
#             print("кадров в логе :", len(ts_map[node]))
#         else:
#             print("список ts_map keys: ", ts_map.keys())

#         asr = transcribe(model, str(audio))
#         print(asr)
#         for s in asr["segments"]:
#             # s["speaker"] = node         # имя файла → ярлык
#             # all_segments.append(s)
#             s["speaker"] = node
#             abs_ts = to_absolute(s["start"], ts_map[node])
#             s["abs_start"] = abs_ts
#             all_segments.append(s)

#     # all_segments.sort(key=lambda x: x["start"])
#     all_segments.sort(key=lambda x: x["abs_start"])

#     txt = segments_to_dialogue(all_segments)
#     out = Path(folder) / "dialogue.txt"
#     out.write_text(txt, "utf-8")
#     print(f"✓ dialogue saved to {out}")


def merge_consecutive_speaker_segments(segments, merge_gap_ms=400):
    """
    segments  – list[dict] отсортированный по 'abs_start'.
    Возвращает новый list[dict] с объединёнными репликами.
    """
    if not segments:
        return []

    merged = [segments[0].copy()]           # стартуем с первого сегмента
    for seg in segments[1:]:
        cur = merged[-1]

        same_speaker = seg["speaker"] == cur["speaker"]
        gap_ms = (seg["abs_start"] - cur["abs_start"]).total_seconds() * 1000 \
                 - (cur["end"] - cur["start"]) * 1000

        if same_speaker and gap_ms <= merge_gap_ms:
            # ── объединяем ─────────────────────────────────────────────
            cur["text"]  += " " + seg["text"].lstrip()
            cur["end"]    = seg["end"]          # относительный конец
            cur["words"] += seg["words"]        # расширяем массив слов
        else:
            merged.append(seg.copy())

    return merged


def multi_track(folder, args, gap_ms=2000):
    model = load_model(args.model, args.language, args.device, args.compute_type)
    ts_map = get_meeting_event_log(folder)

    all_segments = []
    for audio in sorted(Path(folder).glob("*.wav")):
        print("processing audio: ", audio)
        node = id_from_wav(audio)
        asr  = transcribe(model, str(audio))

        # ① precalc разрывы по логам
        gaps_idx = find_log_gaps(ts_map[node], gap_ms)

        for seg in asr["segments"]:
            # ② split по log‑gaps + слово‑тайм‑штампы
            new_segs = split_segment_by_log(seg,
                                             ts_map[node],
                                             gaps_idx,
                                             node)
            for s in new_segs:
                # ③ абсолютное время для дальнейшей сортировки
                s["abs_start"] = abs_time(s["start"], ts_map[node])
                all_segments.append(s)

    all_segments.sort(key=lambda x: x["abs_start"])
    all_segments = merge_consecutive_speaker_segments(all_segments, merge_gap_ms=400)
    txt = segments_to_dialogue(all_segments)
    
    (Path(folder) / "dialogue.txt").write_text(txt, "utf-8")

# ---------- CLI ----------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", help="Файл (один трек) или папка (мульти-трек).",
                   default=r"C:\Users\Evgeniy\Desktop\ss3\zoom_assistant\py-zoom-meeting-sdk\sample_program\out\audio\72611797337_20250714_071633")
                #    default=r"C:\Users\Evgeniy\Desktop\ss3\zoom_assistant\py-zoom-meeting-sdk\sample_program\out\audio\72611797337_20250714_071633 — копия")
                #    default=r"C:\Users\Evgeniy\Desktop\meetings\common\meeting_20250704_044000.wav")
    p.add_argument("--model", default="large")
    p.add_argument("--language", default="ru")
    p.add_argument("--device", default="cpu")
    p.add_argument("--hf_token", default=None,
                   help="HF token для диаризации.")
    p.add_argument("--compute_type", default="float32",
        help="int8 | float32 | int8_float16 | float16 | auto")
    args = p.parse_args()

    inp = Path(args.input)
    if inp.is_dir():
        multi_track(inp, args)
    else:
        single_track(inp, args)
