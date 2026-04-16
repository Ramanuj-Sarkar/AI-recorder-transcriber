# Overlapping Audio Recorder

An audio recording tool that captures microphone input across two overlapping streams, producing a set of one-minute `.wav` files and automatically transcribing them to text using [OpenAI Whisper](https://github.com/openai/whisper).

---

## How It Works

Two recording streams run simultaneously, offset by 30 seconds:

- **Stream A** begins recording immediately when you start a session.
- **Stream B** begins 30 seconds later, covering the same audio windows with a deliberate overlap.

Each stream produces independent, one-minute `.wav` segments. Because the streams are offset, every moment of audio is captured in at least one file from each stream — eliminating the brief gaps that would occur if a single stream had to stop, save, and restart. Stream B serves as a redundant audio backup; Stream A files are the ones transcribed to text.

After recording ends, Whisper runs a transcription pass over all Stream A segments, producing a plain-text file for each one.

---

## Output Structure

```
recordings/
├── audio/
│   ├── Segment_001_A.wav
│   ├── Segment_001_B.wav
│   ├── Segment_002_A.wav
│   ├── Segment_002_B.wav
│   └── ...
└── transcripts/
    ├── Segment_001_Transcription.txt
    ├── Segment_002_Transcription.txt
    └── ...
```

Matching numbers indicate files that cover the same audio window. Stream B `.wav` files are retained as a listening backup but are not transcribed.

---

## Requirements

**Python packages:**

```bash
pip install sounddevice soundfile numpy openai-whisper
```

**System dependency** — PortAudio is required by `sounddevice` for microphone access:

| Platform | Command |
|----------|---------|
| macOS | `brew install portaudio` |
| Ubuntu / Debian | `sudo apt install portaudio19-dev` |
| Windows | No additional steps required |

> **Note:** Whisper will download its model weights on first use. The `small` model is approximately 460 MB.

---

## Usage

```bash
python recorder.py
```

Follow the on-screen prompts:

1. Press **Enter** to begin recording.
2. Press **Enter** again to stop.
3. Transcription runs automatically once recording ends.

### Choosing a Whisper Model

Pass the `--model` flag to select a different model size:

```bash
python recorder.py --model base
```

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `tiny` | ~75 MB | Fastest | Lower |
| `base` | ~145 MB | Fast | Moderate |
| `small` | ~460 MB | Moderate | Good *(default)* |
| `medium` | ~1.5 GB | Slow | Better |
| `large` | ~3 GB | Slowest | Best |

For clear speech recorded close to the microphone, `small` is a practical default. Use `base` if transcription speed is a priority, or `medium`/`large` for challenging audio.

---

## Configuration

Key settings are defined as constants near the top of `recorder.py` and can be adjusted to suit your needs:

| Constant | Default | Description |
|----------|---------|-------------|
| `SAMPLE_RATE` | `44100` | Audio sample rate in Hz |
| `CHANNELS` | `1` | Number of audio channels (1 = mono) |
| `SEGMENT_SECONDS` | `60` | Duration of each recorded segment |
| `STREAM_OFFSET` | `30` | Seconds before Stream B begins |
| `OVERLAP_SECONDS` | `5` | Extra seconds buffered at each segment boundary |
| `WHISPER_MODEL` | `"small"` | Default Whisper model (overridden by `--model`) |
| `OUTPUT_DIR` | `recordings/` | Root output directory |

---

## Notes

- Audio is recorded in **mono** at 44,100 Hz, which is more than sufficient for speech and keeps file sizes manageable.
- Segments are saved as **16-bit PCM WAV**, which writes almost instantaneously and requires no encoding step.
- Transcription runs as a **post-processing pass** after recording completes, so it does not compete with the recording threads for CPU resources during capture.
- Both streams receive a **full, independent copy** of the audio via separate queues — they do not share a buffer, which would otherwise cause each stream to receive only half the audio and play back at double speed.
