"""
mms_streaming_eval.py
======================
Real-time streaming evaluation of the LoRA-fine-tuned MMS-1B-all model on the
MASRI Headset v2 streaming corpus, using growing-buffer chunked CTC with
final-transcript reporting.

Method (v3)
-----------
Earlier attempts failed in two distinct ways:

  (v1) "Chunk-and-emit": run MMS on each chunk independently; emit the
       chunk's text. Produced severe partial-word duplication artifacts at
       chunk boundaries, inflating WER to 52%.
  (v2) "Growing-buffer + suffix delta": run MMS on the full accumulated
       buffer each chunk; emit the new-words suffix. Failed because
       successive decodes of the growing buffer disagree on word
       boundaries (e.g. "poġġietilhom" vs "poġġie tilhom"), breaking the
       word-level common-prefix match and re-emitting hundreds of words
       per clip — WERs of 200-900% (n_words_emitted >> n_words_reference).

This v3 takes the simplest robust approach:

  1. Each chunk's audio is appended to a growing buffer.
  2. The buffer is decoded; the resulting transcript REPLACES the running
     transcript (no per-word emission).
  3. The final transcript at end-of-stream is the streamed output.
  4. Per-word latency is recovered post-hoc using a character-level
     longest-common-prefix between consecutive chunk decodes: a word is
     "committed" in chunk N if it appears in chunk N's transcript at a
     position contained within the LCP of chunks N and N-1 (i.e., a word
     is committed once it stops changing). Words at the unstable tail of
     each chunk's transcript are not yet committed.

This separates the two concerns the earlier versions conflated:
  - The streamed output is just the model's last full decode (no
    fragmentation artifacts, no emission heuristics).
  - The latency measurement is a separate post-hoc analysis on the
    decode history.

Streaming protocol (defended in thesis methodology)
---------------------------------------------------
- 2.0-second audio chunks. Matched with Whisper/Seamless streaming for
  fair cross-model comparison.
- Growing-buffer decoding. Each chunk arrival decodes the full audio
  accumulated so far. MMS's CTC head is monotonic and inference is fast
  enough (>27× realtime) to make this practical.
- Greedy CTC argmax decoding (no beam search) — matches the offline MMS
  evaluation, so streaming-vs-offline WER degradation is attributable to
  the streaming protocol alone.
- Per-word commit latency is defined as the time at which a word's
  position in the decoded transcript becomes stable (no longer revised
  by subsequent chunk decodes), measured against the audio arrival time
  of the chunk in which it stabilised.

References
----------
- Pratap et al. 2024 (JMLR) — MMS architecture and pretraining.
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
from transformers import AutoModelForCTC, AutoProcessor
from jiwer import wer as compute_wer, cer as compute_cer


# ==========================================
# CONFIG
# ==========================================
MODEL_PATH        = r"C:\MMS_v2"
STREAMING_CSV     = r"G:\My Drive\Thesis Project\MASRI_HEADSET_v2\streaming_metadata.csv"
RESULTS_AGG_FILE  = "./mms_streaming_results.csv"
RESULTS_WORD_FILE = "./mms_streaming_words.csv"

LANGUAGE        = "mlt"
SAMPLE_RATE     = 16000
CHUNK_SIZE_SEC  = 2.0


# ==========================================
# HELPERS
# ==========================================
def normalize_text(text):
    """Same normaliser used during training and offline eval."""
    text = text.lower()
    text = re.sub(r"[^\w\s'-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def decode_audio_buffer(audio_buffer, model, processor, device, dtype):
    """Run MMS forward pass on the full accumulated audio buffer."""
    inputs = processor(
        audio_buffer,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    ).to(device)
    if dtype == torch.float16:
        inputs["input_values"] = inputs["input_values"].to(torch.float16)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(pred_ids)[0]
    return normalize_text(text)


def char_common_prefix_len(a, b):
    """Character-level longest common prefix length."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def stream_one_clip(audio, model, processor, device, dtype):
    """
    Growing-buffer streaming with final-transcript reporting and post-hoc
    per-word latency assignment.

    Returns:
        committed_words: list of {word, audio_arrival_sec, commit_time_sec}.
                         Order matches the final transcript.
        final_text:      The model's last full decode (the streamed output).
        total_proc_sec:  Wall-clock time inside the streaming loop.
    """
    chunk_samples = int(CHUNK_SIZE_SEC * SAMPLE_RATE)

    # Per-chunk decode history: (chunk_audio_arrival_sec, chunk_commit_walltime, transcript)
    history = []

    accumulated_audio = np.array([], dtype=np.float32)
    proc_start_wall = time.time()
    audio_pos_samples = 0

    while audio_pos_samples < len(audio):
        chunk = audio[audio_pos_samples : audio_pos_samples + chunk_samples]
        audio_pos_samples += len(chunk)
        audio_arrival_sec = audio_pos_samples / SAMPLE_RATE

        accumulated_audio = np.concatenate([accumulated_audio, chunk])

        try:
            transcript = decode_audio_buffer(
                accumulated_audio, model, processor, device, dtype
            )
        except Exception as e:
            print(f"      WARN: chunk forward raised {type(e).__name__}: {e}",
                  flush=True)
            continue

        wall_now = time.time()
        history.append((audio_arrival_sec, wall_now - proc_start_wall, transcript))

    total_proc_sec = time.time() - proc_start_wall

    if not history:
        return [], "", total_proc_sec

    # Final transcript is the last decode
    final_text = history[-1][2]
    final_words = final_text.split()

    # Per-word commit time via character-level LCP across chunks:
    # walk through chunks in order; a word in the final transcript is
    # considered "committed" at the earliest chunk whose decode and the
    # final decode agree on the character prefix up to that word's end.
    committed_words = []
    final_word_end_chars = []
    cumulative = 0
    for w in final_words:
        cumulative += len(w) + 1     # +1 for the trailing space (or imaginary trailing space for the last word)
        final_word_end_chars.append(cumulative)

    for wi, word in enumerate(final_words):
        word_end_char = final_word_end_chars[wi]
        # Find the first chunk whose decode agrees with final_text up to
        # (at least) this word's end-of-word character position.
        commit_audio_arrival = history[-1][0]   # default: end of stream
        commit_walltime      = history[-1][1]
        for (audio_t, wall_t, transcript) in history:
            lcp = char_common_prefix_len(transcript, final_text)
            if lcp >= word_end_char:
                commit_audio_arrival = audio_t
                commit_walltime      = wall_t
                break

        committed_words.append({
            "word":              word,
            "audio_arrival_sec": commit_audio_arrival,
            "commit_time_sec":   commit_walltime,
        })

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

    print(f"Loading MMS from {MODEL_PATH}...", flush=True)
    t_load = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, target_lang=LANGUAGE)
    model = AutoModelForCTC.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        ignore_mismatched_sizes=True,
    ).to(device)
    model.eval()
    print(f"  Model load took {time.time() - t_load:.1f}s", flush=True)
    print(f"  Precision: {dtype}; device: {device}", flush=True)

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

    print(f'  Sample final transcript: "{_final[:200]}..."', flush=True)
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

        # streamed_text is already normalized inside decode_audio_buffer; safe-renormalize
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
              f"audio={audio_dur:5.1f}s  proc={proc_sec:5.2f}s  rtf={rtf:.3f}  "
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
    print(f"MMS STREAMING RESULTS (growing-buffer CTC, chunk={CHUNK_SIZE_SEC}s)")
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