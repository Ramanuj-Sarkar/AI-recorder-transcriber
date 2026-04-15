"""
Overlapping audio recorder with Whisper transcription.

Two streams record simultaneously, offset by 30 seconds, each producing
60-second .wav files. After recording, Stream A files are transcribed.
Stream B files serve as audio backup covering the same windows.

Dependencies:
    pip install sounddevice soundfile numpy openai-whisper
"""

import threading
import queue
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_RATE       = 44100         # Hz
CHANNELS          = 1             # Mono is fine for voice/speech
SEGMENT_SECONDS   = 60            # Length of each audio file
STREAM_OFFSET     = 30            # Seconds before Stream B starts
OVERLAP_SECONDS   = 5             # Extra seconds recorded at segment end
                                  # to guarantee no gap at the join point
WHISPER_MODEL     = "small"       # tiny / base / small / medium / large
OUTPUT_DIR        = Path("recordings")
AUDIO_SUBDIR      = OUTPUT_DIR / "audio"
TRANSCRIPT_SUBDIR = OUTPUT_DIR / "transcripts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dirs():
    AUDIO_SUBDIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_SUBDIR.mkdir(parents=True, exist_ok=True)


def samples_for(seconds: float) -> int:
    return int(seconds * SAMPLE_RATE)


def save_as_wav(pcm: np.ndarray, path: Path):
    """Write raw PCM directly to a .wav file."""
    sf.write(str(path), pcm, SAMPLE_RATE, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Audio capture with per-stream fan-out
# ---------------------------------------------------------------------------

class AudioCapture:
    """
    Continuously captures audio from the microphone and fans out each chunk
    to a separate queue per registered stream. This ensures every stream
    receives a full, independent copy of the audio rather than competing
    for chunks from a shared buffer (which would cause sped-up playback).
    """

    def __init__(self):
        self._queues: list = []
        self._stream = None
        self._lock = threading.Lock()

    def add_stream_queue(self) -> queue.Queue:
        """Register a new consumer and return its dedicated queue."""
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._queues.append(q)
        return q

    def start(self):
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=1024,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"  [audio status] {status}")
        chunk = indata.copy()
        with self._lock:
            for q in self._queues:
                q.put(chunk)


def read_seconds(buf: queue.Queue, seconds: float, stop_event: threading.Event) -> np.ndarray:
    """
    Drain `seconds` worth of audio from a stream's dedicated queue.
    Returns early if stop_event is set AND the queue empties.
    """
    needed = samples_for(seconds)
    chunks = []
    collected = 0

    while collected < needed:
        if stop_event.is_set() and buf.empty():
            break
        try:
            chunk = buf.get(timeout=0.05)
            chunks.append(chunk)
            collected += len(chunk)
        except queue.Empty:
            continue

    if not chunks:
        return np.zeros((0, CHANNELS), dtype="float32")

    audio = np.concatenate(chunks, axis=0)
    return audio[:needed]


# ---------------------------------------------------------------------------
# Stream worker
# ---------------------------------------------------------------------------

def stream_worker(
    name: str,
    buf: queue.Queue,
    stop_event: threading.Event,
    segment_counter: list,
    counter_lock: threading.Lock,
    segment_queue: queue.Queue,
    delay_seconds: float = 0.0,
):
    """
    Records overlapping 60-second segments from a dedicated audio queue.
    Files are named Segment_NNN_A.wav or Segment_NNN_B.wav.
    """
    stream_tag = "A" if name == "StreamA" else "B"

    if delay_seconds > 0:
        # Wait, but honour stop_event so we don't block forever
        stop_event.wait(timeout=delay_seconds)
        if stop_event.is_set():
            return

    seg_index = 1

    while not stop_event.is_set():
        record_secs = SEGMENT_SECONDS + OVERLAP_SECONDS
        audio = read_seconds(buf, record_secs, stop_event)

        if len(audio) == 0:
            break

        # Trim to clean segment length (overlap was just insurance)
        audio = audio[:samples_for(SEGMENT_SECONDS)]

        filename = f"Segment_{seg_index:03d}_{stream_tag}.wav"
        path = AUDIO_SUBDIR / filename

        print(f"  Saving {filename} …")
        save_as_wav(audio, path)
        segment_queue.put((path, stream_tag))

        with counter_lock:
            segment_counter[0] += 1

        seg_index += 1


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_stream_a(segment_queue: queue.Queue, done_event: threading.Event):
    """
    Transcribes Stream A (.wav) files after recording finishes.
    Stream B files are left as the audio backup.
    """
    print("\nLoading Whisper model …")
    model = whisper.load_model(WHISPER_MODEL)
    print("Whisper ready.\n")

    while not (done_event.is_set() and segment_queue.empty()):
        try:
            path, stream_tag = segment_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if stream_tag != "A":
            continue   # Only transcribe Stream A; Stream B is the audio backup

        stem = path.stem   # e.g. "Segment_001_A"
        transcript_name = stem.replace("_A", "_Transcription") + ".txt"
        transcript_path = TRANSCRIPT_SUBDIR / transcript_name

        print(f"  Transcribing {path.name} …")
        result = model.transcribe(str(path), language=None, fp16=False)
        text = result["text"].strip()

        transcript_path.write_text(text, encoding="utf-8")
        print(f"  -> {transcript_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Overlapping audio recorder")
    parser.add_argument(
        "--model",
        default=WHISPER_MODEL,
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: small)",
    )
    args = parser.parse_args()

    ensure_dirs()

    print("=" * 60)
    print("  Overlapping Audio Recorder")
    print(f"  Output : {OUTPUT_DIR.resolve()}")
    print(f"  Whisper: {args.model} model")
    print(f"  Streams: A starts immediately, B starts after {STREAM_OFFSET}s")
    print("=" * 60)
    input("\nPress ENTER to start recording ...")
    print(f"\nRecording started at {datetime.now().strftime('%H:%M:%S')}")
    print("Press ENTER again to stop.\n")

    capture = AudioCapture()

    # Each stream gets its own dedicated queue -- no sharing, no competition
    buf_a = capture.add_stream_queue()
    buf_b = capture.add_stream_queue()

    capture.start()

    stop_event    = threading.Event()
    counter_lock  = threading.Lock()
    segment_count = [0]
    segment_queue: queue.Queue = queue.Queue()

    thread_a = threading.Thread(
        target=stream_worker,
        kwargs=dict(
            name="StreamA",
            buf=buf_a,
            stop_event=stop_event,
            segment_counter=segment_count,
            counter_lock=counter_lock,
            segment_queue=segment_queue,
            delay_seconds=0.0,
        ),
        daemon=True,
    )
    thread_b = threading.Thread(
        target=stream_worker,
        kwargs=dict(
            name="StreamB",
            buf=buf_b,
            stop_event=stop_event,
            segment_counter=segment_count,
            counter_lock=counter_lock,
            segment_queue=segment_queue,
            delay_seconds=float(STREAM_OFFSET),
        ),
        daemon=True,
    )

    thread_a.start()
    thread_b.start()

    # Block until the user presses ENTER
    input()
    print("\nStopping recording ...")
    stop_event.set()

    thread_a.join()
    thread_b.join()
    capture.stop()

    total = segment_count[0]
    print(f"Recording complete. {total} audio file(s) saved to {AUDIO_SUBDIR}/")

    # --- Transcription pass ---
    transcription_done = threading.Event()
    transcription_done.set()   # queue is fully populated; signal done immediately

    print("\nStarting transcription pass ...")
    transcribe_stream_a(segment_queue, transcription_done)

    print("\nAll done.")
    print(f"  Audio files  : {AUDIO_SUBDIR}/")
    print(f"  Transcripts  : {TRANSCRIPT_SUBDIR}/")


if __name__ == "__main__":
    main()
