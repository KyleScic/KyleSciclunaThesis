"""
whisper_streaming_eval.py
==========================
Real-time streaming evaluation of the LoRA-fine-tuned Whisper-Medium model
on the MASRI Headset v2 streaming corpus.

Method
------
LocalAgreement-2 algorithm (Liu et al. 2020, applied to Whisper by
Macháček, Dabre & Bojar, 2023 IJCNLP-AACL System Demos paper). Uses the
upstream `ufal/whisper_streaming` reference implementation as a library,
backed by faster-whisper (CTranslate2) for low-latency inference.

Streaming protocol (defended in thesis methodology)
---------------------------------------------------
- 2.0-second audio chunks. Raised from the upstream library's default of 1.0s
  after a first attempt at 1.0s showed RTF ~7-9× on the target hardware
  (RTX 3050); doubling the chunk size halves the number of re-decodes per
  clip in LocalAgreement-2.
- Greedy decoding (beam_size = 1) for streaming. LocalAgreement-2 re-decodes
  the growing audio buffer on every chunk, so each token is generated dozens
  of times across the clip's lifetime; greedy decoding removes the 5× beam-
  search multiplier. The offline evaluation in this thesis uses beam_size=5
  for best-effort transcription quality — the streaming chapter therefore
  characterises the accuracy-versus-latency tradeoff between the two regimes
  rather than a single "best result" number.
- VAD filter disabled. The MASRI streaming clips are pre-segmented and contain
  speech throughout (with controlled 400ms silences between member utterances);
  VAD adds latency and can drop content without benefit here.
- The same chunk size and decoding settings are applied to all three model
  streaming scripts (Whisper, MMS, Seamless) so the cross-model comparison
  remains fair.
"""

# ============================================================================
# CUDA DLL DISCOVERY — must happen BEFORE any CUDA-dependent import
# ============================================================================
import os

_venv_site = r"C:\Users\kylex\PycharmProjects\GenerateAudio\.venv\Lib\site-packages"
for _sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
    _p = os.path.join(_venv_site, *_sub.split("/"))
    if hasattr(os, "add_dll_directory") and os.path.exists(_p):
        os.add_dll_directory(_p)
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")
        print(f"  added DLL dir: {_p}", flush=True)
    else:
        print(f"  MISSING DLL dir: {_p}", flush=True)

# ============================================================================
# Standard imports
# ============================================================================
import csv
import re
import time
import numpy as np
import librosa
from jiwer import wer as compute_wer, cer as compute_cer
from whisper_online import FasterWhisperASR, OnlineASRProcessor


# ==========================================
# CONFIG
# ==========================================
MODEL_PATH        = r"C:\Whisper_CT2_v2"
STREAMING_CSV     = r"G:\My Drive\Thesis Project\MASRI_HEADSET_v2\streaming_metadata.csv"
RESULTS_AGG_FILE  = "./whisper_streaming_results.csv"
RESULTS_WORD_FILE = "./whisper_streaming_words.csv"

LANGUAGE       = "mt"
SAMPLE_RATE    = 16000
CHUNK_SIZE_SEC = 2.0
BEAM_SIZE      = 1            # greedy decoding for streaming


# ==========================================
# HELPERS
# ==========================================
def normalize_text(text):
    """Same normaliser as training-time and offline eval."""
    text = text.lower()
    text = re.sub(r"[^\w\s'-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def stream_one_clip(audio, asr, online):
    """
    Feed `audio` (1-D float32 array @16 kHz) to the OnlineASRProcessor in
    CHUNK_SIZE_SEC-second chunks, logging every committed word with its
    commit time and the audio arrival time of the chunk that triggered the
    commit.
    """
    online.init()
    chunk_samples = int(CHUNK_SIZE_SEC * SAMPLE_RATE)

    committed_words = []
    proc_start_wall = time.time()
    audio_pos_samples = 0

    while audio_pos_samples < len(audio):
        chunk = audio[audio_pos_samples : audio_pos_samples + chunk_samples]
        audio_pos_samples += chunk_samples
        audio_arrival_sec = audio_pos_samples / SAMPLE_RATE

        online.insert_audio_chunk(chunk)
        try:
            start_ts, end_ts, committed_text = online.process_iter()
        except Exception as e:
            print(f"      WARN: process_iter raised {type(e).__name__}: {e}", flush=True)
            continue

        wall_now = time.time()
        if committed_text and committed_text.strip():
            for word in committed_text.strip().split():
                committed_words.append({
                    "word":              word,
                    "audio_arrival_sec": audio_arrival_sec,
                    "commit_time_sec":   wall_now - proc_start_wall,
                })

    try:
        start_ts, end_ts, final_text = online.finish()
    except Exception as e:
        print(f"      WARN: finish() raised {type(e).__name__}: {e}", flush=True)
        final_text = ""
    wall_now = time.time()
    if final_text and final_text.strip():
        final_audio_sec = len(audio) / SAMPLE_RATE
        for word in final_text.strip().split():
            committed_words.append({
                "word":              word,
                "audio_arrival_sec": final_audio_sec,
                "commit_time_sec":   wall_now - proc_start_wall,
            })

    total_proc_sec = time.time() - proc_start_wall
    return committed_words, total_proc_sec


# ==========================================
# MAIN
# ==========================================
def main():
    with open(STREAMING_CSV, "r", encoding="utf-8") as f:
        clips = list(csv.DictReader(f))
    print(f"Loaded {len(clips)} streaming clips from {STREAMING_CSV}\n", flush=True)

    print(f"Loading FasterWhisperASR from {MODEL_PATH}...", flush=True)
    t_load_start = time.time()
    asr = FasterWhisperASR(
        modelsize=None,
        lan=LANGUAGE,
        cache_dir=None,
        model_dir=MODEL_PATH,
    )
    print(f"  Model load took {time.time() - t_load_start:.1f}s", flush=True)

    # Streaming-mode decoding settings.
    #
    # IMPORTANT: beam_size must NOT be placed in asr.transcribe_kargs — the
    # FasterWhisperASR.transcribe() method already passes beam_size= explicitly
    # to faster_whisper.WhisperModel.transcribe(), and any value in
    # transcribe_kargs becomes a duplicate-kwarg TypeError on every chunk.
    # The library instead reads `asr.beam_size` if that attribute exists.
    asr.transcribe_kargs["vad_filter"] = False
    try:
        asr.beam_size = BEAM_SIZE
    except Exception:
        pass
    print(f"  VAD disabled; beam_size={BEAM_SIZE} (greedy)", flush=True)

    online = OnlineASRProcessor(asr)

    # Warm-up — 1 kHz tone is unambiguously not silence
    print(f"Warming up (one {CHUNK_SIZE_SEC}s 1kHz tone chunk)...", flush=True)
    t = np.arange(int(CHUNK_SIZE_SEC * SAMPLE_RATE)) / SAMPLE_RATE
    dummy = (0.3 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    t_warm_start = time.time()
    _ = stream_one_clip(dummy, asr, online)
    print(f"Warm-up complete in {time.time() - t_warm_start:.1f}s.\n", flush=True)

    # ==========================================
    # SANITY CHECK — abort if streaming pipeline is broken
    # ==========================================
    # Run a single real clip first. If no words are committed (the failure
    # mode we've seen multiple times), abort here rather than after a full
    # 1-2 hour run produces an empty CSV.
    print("Sanity check: streaming first clip only...", flush=True)
    first = clips[0]
    _audio, _ = librosa.load(first["clip_path"], sr=SAMPLE_RATE)
    _audio = _audio.astype(np.float32)
    _words, _proc = stream_one_clip(_audio, asr, online)
    _audio_dur = len(_audio) / SAMPLE_RATE
    print(f"  -> {len(_words)} words emitted in {_proc:.1f}s "
          f"(audio={_audio_dur:.1f}s, rtf={_proc/_audio_dur:.2f})", flush=True)

    if len(_words) == 0:
        print("\n  FAIL: zero words emitted on sanity check.")
        print("  The streaming pipeline is broken — aborting before the full run.")
        print("  Check the WARN: lines above for the root cause.\n")
        return

    print(f'  Sample committed text: "{" ".join(w["word"] for w in _words[:15])}..."', flush=True)
    print(f"  Sanity check passed.  Continuing with the full corpus.\n", flush=True)

    # ==========================================
    # MAIN STREAMING LOOP
    # ==========================================
    # Note: the sanity-check clip is also re-streamed below as part of the
    # main loop so it appears in the output CSV with everything else. (Cheap
    # — adds one extra clip's worth of compute to a multi-hour run.)
    agg_rows  = []
    word_rows = []

    for i, row in enumerate(clips):
        clip_id     = row["clip_id"]
        clip_path   = row["clip_path"]
        reference   = normalize_text(row["reference"])
        audio_dur   = float(row["duration_sec"])

        try:
            audio, _ = librosa.load(clip_path, sr=SAMPLE_RATE)
        except Exception as e:
            print(f"  SKIP unreadable clip {clip_id}: {e}", flush=True)
            continue
        audio = audio.astype(np.float32)

        committed_words, proc_sec = stream_one_clip(audio, asr, online)

        streamed_text = " ".join(w["word"] for w in committed_words)
        streamed_text = normalize_text(streamed_text)
        word_latencies = [
            max(0.0, w["commit_time_sec"] - w["audio_arrival_sec"])
            for w in committed_words
        ]

        if reference.strip():
            clip_wer = compute_wer([reference], [streamed_text]) if streamed_text else 1.0
            clip_cer = compute_cer([reference], [streamed_text]) if streamed_text else 1.0
        else:
            clip_wer = clip_cer = float("nan")
        rtf = proc_sec / audio_dur if audio_dur > 0 else float("nan")
        median_lat = float(np.median(word_latencies)) if word_latencies else float("nan")
        p95_lat    = float(np.percentile(word_latencies, 95)) if word_latencies else float("nan")

        agg_rows.append({
            "clip_id":            clip_id,
            "n_words_committed":  len(committed_words),
            "audio_sec":          round(audio_dur, 3),
            "proc_sec":           round(proc_sec, 3),
            "rtf":                round(rtf, 3),
            "median_word_lat":    round(median_lat, 3),
            "p95_word_lat":       round(p95_lat, 3),
            "streaming_wer":      round(clip_wer, 4),
            "streaming_cer":      round(clip_cer, 4),
            "reference":          reference,
            "streamed":           streamed_text,
        })
        for w in committed_words:
            word_rows.append({
                "clip_id":           clip_id,
                "word":              w["word"],
                "audio_arrival_sec": round(w["audio_arrival_sec"], 3),
                "commit_time_sec":   round(w["commit_time_sec"], 3),
                "word_latency_sec":  round(max(0.0, w["commit_time_sec"] - w["audio_arrival_sec"]), 3),
            })

        print(f"  [{i+1:3d}/{len(clips)}] {clip_id}  "
              f"audio={audio_dur:5.1f}s  proc={proc_sec:5.1f}s  rtf={rtf:.2f}  "
              f"WER={clip_wer*100:5.1f}%  median_lat={median_lat:.2f}s  p95_lat={p95_lat:.2f}s",
              flush=True)

        # Incremental save every 5 clips so a mid-run crash doesn't lose everything
        if (i + 1) % 5 == 0:
            with open(RESULTS_AGG_FILE, "w", encoding="utf-8", newline="") as f:
                w_ = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
                w_.writeheader()
                w_.writerows(agg_rows)
            if word_rows:
                with open(RESULTS_WORD_FILE, "w", encoding="utf-8", newline="") as f:
                    w_ = csv.DictWriter(f, fieldnames=list(word_rows[0].keys()))
                    w_.writeheader()
                    w_.writerows(word_rows)

    # ==========================================
    # CORPUS-LEVEL AGGREGATE
    # ==========================================
    refs_for_wer = [r["reference"] for r in agg_rows if r["reference"].strip()]
    sts_for_wer  = [r["streamed"]  for r in agg_rows if r["reference"].strip()]
    corpus_wer = compute_wer(refs_for_wer, sts_for_wer) if refs_for_wer else float("nan")
    corpus_cer = compute_cer(refs_for_wer, sts_for_wer) if refs_for_wer else float("nan")
    all_lats   = [w["word_latency_sec"] for w in word_rows]

    print(f"\n{'='*60}")
    print(f"WHISPER STREAMING RESULTS (LocalAgreement-2, chunk={CHUNK_SIZE_SEC}s, beam={BEAM_SIZE})")
    print(f"{'='*60}")
    print(f"  Clips streamed:        {len(agg_rows)}")
    print(f"  Total words emitted:   {len(word_rows)}")
    print(f"  Corpus streaming WER:  {corpus_wer*100:5.2f}%")
    print(f"  Corpus streaming CER:  {corpus_cer*100:5.2f}%")
    if all_lats:
        print(f"  Word latency — median: {np.median(all_lats):.3f}s")
        print(f"  Word latency — mean:   {np.mean(all_lats):.3f}s")
        print(f"  Word latency — P95:    {np.percentile(all_lats, 95):.3f}s")
    rtfs = [r["rtf"] for r in agg_rows if not np.isnan(r["rtf"])]
    if rtfs:
        print(f"  Mean RTF:              {np.mean(rtfs):.3f}")
    print(f"{'='*60}")

    if agg_rows:
        with open(RESULTS_AGG_FILE, "w", encoding="utf-8", newline="") as f:
            w_ = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            w_.writeheader()
            w_.writerows(agg_rows)
        print(f"\nPer-clip results:  {RESULTS_AGG_FILE}")

    if word_rows:
        with open(RESULTS_WORD_FILE, "w", encoding="utf-8", newline="") as f:
            w_ = csv.DictWriter(f, fieldnames=list(word_rows[0].keys()))
            w_.writeheader()
            w_.writerows(word_rows)
        print(f"Per-word results:  {RESULTS_WORD_FILE}")
    else:
        print(f"\nNo words were committed across any clip — check upstream errors before re-running.")


if __name__ == "__main__":
    main()