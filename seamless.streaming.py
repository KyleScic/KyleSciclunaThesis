"""
seamless_streaming_eval.py
===========================
Real-time streaming evaluation of the LoRA-fine-tuned SeamlessM4T v2-Large
model on the MASRI Headset v2 streaming corpus.

Method (v3) — per-chunk independent decoding
---------------------------------------------
SeamlessM4T v2 was trained as an utterance-level speech-to-text translation
model rather than for long-form transcription. Two earlier streaming
approaches were attempted:

  (v1) LocalAgreement-2 on a growing audio buffer. Failed because greedy
       Seamless decodes produce shifting word boundaries between chunks;
       word-level prefix matching committed very few words (median ~10/clip)
       and WER reached 85-95%.

  (v2) Growing-buffer with final-transcript replacement (the protocol that
       worked for MMS). Failed because Seamless's decoder emits an early
       EOS token when growing-buffer audio exceeds the typical utterance
       length the model was trained on. Streamed transcripts were ~40
       characters vs ~500-character references; WER remained 85-95%.

This v3 implements per-chunk independent decoding: each 2-second chunk is
transcribed as a standalone short utterance — which is exactly what
Seamless was trained to do — and the chunk outputs are concatenated to
form the streamed transcript. Latency per word is therefore bounded by
the chunk size (each word is committed in the chunk it occurred in,
with no need for cross-chunk agreement).

Tradeoffs (acknowledged in thesis methodology)
-----------------------------------------------
- The model loses cross-chunk context. Words spoken at the boundary of two
  chunks may be split, missed, or duplicated. This is the cost of using a
  model not designed for streaming.
- Output WER is expected to be substantially higher than the offline
  Seamless WER (~29%) because of these boundary effects, but should be
  meaningfully better than the failed growing-buffer attempts (>85%).
- Per-word latency is the chunk duration plus the per-chunk decode time
  (the word is committed at the chunk's audio_arrival_sec).

Protocol parameters
-------------------
- 2.0-second audio chunks (matched with Whisper/MMS).
- Greedy decoding (num_beams = 1) — matches Whisper streaming and reduces
  per-chunk decode time.
- no_repeat_ngram_size = 3, repetition_penalty = 1.2 — carried from
  offline Seamless eval to prevent generation loops on short chunks.
- VAD disabled.
"""

# ============================================================================
# CUDA DLL DISCOVERY
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
import torch
from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor
from jiwer import wer as compute_wer, cer as compute_cer


# ==========================================
# CONFIG
# ==========================================
MODEL_PATH        = r"C:\Seamless_v2"
STREAMING_CSV     = r"G:\My Drive\Thesis Project\MASRI_HEADSET_v2\streaming_metadata.csv"
RESULTS_AGG_FILE  = "./seamless_streaming_results.csv"
RESULTS_WORD_FILE = "./seamless_streaming_words.csv"

LANGUAGE             = "mlt"
SAMPLE_RATE          = 16000
CHUNK_SIZE_SEC       = 2.0
NUM_BEAMS            = 1
NO_REPEAT_NGRAM_SIZE = 3
REPETITION_PENALTY   = 1.2
GENERATION_MAX_LEN   = 64       # per-chunk: short utterance, small cap


# ==========================================
# HELPERS
# ==========================================
def normalize_text(text):
    """Same normaliser used during training and offline eval."""
    text = text.lower()
    text = re.sub(r"[^\w\s'-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def decode_chunk(chunk_audio, model, processor, device, dtype):
    """Greedy generation on a single short chunk."""
    inputs = processor(
        audio=chunk_audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    ).to(device)
    inputs["input_features"] = inputs["input_features"].to(dtype)

    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            tgt_lang=LANGUAGE,
            num_beams=NUM_BEAMS,
            max_new_tokens=GENERATION_MAX_LEN,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            repetition_penalty=REPETITION_PENALTY,
        )

    vocab_size = processor.tokenizer.vocab_size
    out = output_tokens[0].cpu().numpy()
    out = np.where((out >= 0) & (out < vocab_size), out, processor.tokenizer.pad_token_id)

    text = processor.tokenizer.decode(out, skip_special_tokens=True)
    return normalize_text(text)


def stream_one_clip(audio, model, processor, device, dtype):
    """
    Per-chunk independent streaming.

    Each 2-second chunk is transcribed in isolation; the resulting words
    are committed with the chunk's audio_arrival_sec as their commit
    audio time and the wall-clock time at end-of-chunk as their commit
    walltime.

    Returns:
        committed_words: list of {word, audio_arrival_sec, commit_time_sec}.
        final_text:      Concatenation of all chunk outputs.
        total_proc_sec:  Wall-clock time inside the streaming loop.
    """
    chunk_samples = int(CHUNK_SIZE_SEC * SAMPLE_RATE)

    committed_words = []
    proc_start_wall = time.time()
    audio_pos_samples = 0
    all_chunk_texts = []

    while audio_pos_samples < len(audio):
        chunk = audio[audio_pos_samples : audio_pos_samples + chunk_samples]
        audio_pos_samples += len(chunk)
        audio_arrival_sec = audio_pos_samples / SAMPLE_RATE

        # Skip tiny tail chunks (less than 0.25s) — the model produces
        # garbage on such short inputs
        if len(chunk) < int(0.25 * SAMPLE_RATE):
            continue

        try:
            chunk_text = decode_chunk(chunk, model, processor, device, dtype)
        except Exception as e:
            print(f"      WARN: chunk decode raised {type(e).__name__}: {e}",
                  flush=True)
            continue

        wall_now = time.time()
        if chunk_text:
            all_chunk_texts.append(chunk_text)
            for word in chunk_text.split():
                committed_words.append({
                    "word":              word,
                    "audio_arrival_sec": audio_arrival_sec,
                    "commit_time_sec":   wall_now - proc_start_wall,
                })

    total_proc_sec = time.time() - proc_start_wall
    final_text = " ".join(all_chunk_texts)
    return committed_words, final_text, total_proc_sec


# ==========================================
# MAIN
# ==========================================
def main():
    with open(STREAMING_CSV, "r", encoding="utf-8") as f:
        clips = list(csv.DictReader(f))
    print(f"Loaded {len(clips)} streaming clips from {STREAMING_CSV}\n", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required. Reinstall PyTorch with CUDA support.")
    device = "cuda"
    dtype  = torch.float16

    print(f"Loading SeamlessM4T v2 from {MODEL_PATH}...", flush=True)
    t_load = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    print(f"  Model load took {time.time() - t_load:.1f}s", flush=True)
    print(f"  Precision: {dtype}; device: {device}", flush=True)
    print(f"  Decoding: per-chunk greedy (num_beams={NUM_BEAMS}); "
          f"no_repeat_ngram={NO_REPEAT_NGRAM_SIZE}; rep_penalty={REPETITION_PENALTY}",
          flush=True)

    # Warm-up
    print(f"Warming up (one {CHUNK_SIZE_SEC}s 1kHz tone)...", flush=True)
    t = np.arange(int(CHUNK_SIZE_SEC * SAMPLE_RATE)) / SAMPLE_RATE
    dummy = (0.3 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    t_warm = time.time()
    _ = stream_one_clip(dummy, model, processor, device, dtype)
    print(f"Warm-up complete in {time.time() - t_warm:.1f}s.\n", flush=True)

    # ==========================================
    # SANITY CHECK
    # ==========================================
    print("Sanity check: streaming first clip only...", flush=True)
    first = clips[0]
    _audio, _ = librosa.load(first["clip_path"], sr=SAMPLE_RATE)
    _audio = _audio.astype(np.float32)
    _words, _final, _proc = stream_one_clip(_audio, model, processor, device, dtype)
    _audio_dur = len(_audio) / SAMPLE_RATE
    print(f"  -> {len(_words)} words emitted in {_proc:.1f}s "
          f"(audio={_audio_dur:.1f}s, rtf={_proc/_audio_dur:.3f})", flush=True)

    if len(_words) == 0:
        print("\n  FAIL: zero words emitted on sanity check.")
        print("  Check WARN: lines above. Aborting before full run.\n")
        return

    print(f'  Sample committed text: "{_final[:200]}..."', flush=True)
    print(f"  Sanity check passed. Continuing with the full corpus.\n", flush=True)

    # ==========================================
    # MAIN LOOP
    # ==========================================
    agg_rows  = []
    word_rows = []

    for i, row in enumerate(clips):
        clip_id   = row["clip_id"]
        clip_path = row["clip_path"]
        reference = normalize_text(row["reference"])
        audio_dur = float(row["duration_sec"])

        try:
            audio, _ = librosa.load(clip_path, sr=SAMPLE_RATE)
        except Exception as e:
            print(f"  SKIP unreadable clip {clip_id}: {e}", flush=True)
            continue
        audio = audio.astype(np.float32)

        committed_words, streamed_text, proc_sec = stream_one_clip(
            audio, model, processor, device, dtype
        )

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
        rtf        = proc_sec / audio_dur if audio_dur > 0 else float("nan")
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
              f"audio={audio_dur:5.1f}s  proc={proc_sec:6.1f}s  rtf={rtf:.2f}  "
              f"WER={clip_wer*100:5.1f}%  median_lat={median_lat:.2f}s  p95_lat={p95_lat:.2f}s",
              flush=True)

        # Incremental save every 5 clips
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
    # CORPUS AGGREGATE
    # ==========================================
    refs_for_wer = [r["reference"] for r in agg_rows if r["reference"].strip()]
    sts_for_wer  = [r["streamed"]  for r in agg_rows if r["reference"].strip()]
    corpus_wer = compute_wer(refs_for_wer, sts_for_wer) if refs_for_wer else float("nan")
    corpus_cer = compute_cer(refs_for_wer, sts_for_wer) if refs_for_wer else float("nan")
    all_lats   = [w["word_latency_sec"] for w in word_rows]

    print(f"\n{'='*60}")
    print(f"SEAMLESS STREAMING RESULTS (per-chunk decode, chunk={CHUNK_SIZE_SEC}s, beam={NUM_BEAMS})")
    print(f"{'='*60}")
    print(f"  Clips streamed:        {len(agg_rows)}")
    print(f"  Total words emitted:   {len(word_rows)}")
    print(f"  Corpus streaming WER:  {corpus_wer*100:5.2f}%")
    print(f"  Corpus streaming CER:  {corpus_cer*100:5.2f}%")
    if all_lats:
        print(f"  Word latency - median: {np.median(all_lats):.3f}s")
        print(f"  Word latency - mean:   {np.mean(all_lats):.3f}s")
        print(f"  Word latency - P95:    {np.percentile(all_lats, 95):.3f}s")
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


if __name__ == "__main__":
    main()