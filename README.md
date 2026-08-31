# sinhala-asr

An offline Sinhala speech recognition model. 300M parameters, int8-quantised
ONNX, built to run on a phone: the exported graph transcribes 16 kHz Sinhala
speech at well under real time on mobile-class CPUs.

This is the model behind [Moonstone](https://deshan.dev/projects/moonstone),
an offline Sinhala voice keyboard for Android. The full story of how it was
built — and what failed on the way — is in the
[technical write-up](https://deshan.dev/projects/moonstone/tech/).

## What it is

- **Encoder:** Meta's [omniASR-W2V-300M](https://huggingface.co/facebook/omniASR-W2V-300M)
  (Apache-2.0), used frozen. 24 transformer layers, 1024-dim.
- **Head:** a Sinhala-only CTC output layer over 105 symbols, trained on
  ~185k utterances of transcribed Sinhala read speech
  ([OpenSLR SLR52](https://openslr.org/52/)).
- **Export:** encoder and head fused into one ONNX graph, dynamically
  quantised to int8 (MatMul/Gemm). int8 output is byte-identical to fp32 on
  the test set.

The model takes normalised 16 kHz mono audio and emits per-frame log
probabilities over the symbol table (50 frames per second of audio). It is an
acoustic model only: no language model is included, and greedy decoding is
what the numbers below call "greedy".

## Files

| file | size | |
|---|---|---|
| `sinhala_asr_int8.onnx` | 0.8 MB | the graph (in this repo) |
| `weights.bin` | 342 MB | external weight data ([release asset](../../releases)) |
| `vocab.json` | 1 KB | 105 output symbols; index 0 is the CTC blank (in this repo) |

Download `weights.bin` from the release and place it next to the `.onnx` file —
onnxruntime finds it by relative path.

## Usage

```python
import json
import numpy as np
import onnxruntime as ort
import soundfile as sf

vocab = json.load(open("vocab.json", encoding="utf-8"))
sess = ort.InferenceSession("sinhala_asr_int8.onnx")

audio, sr = sf.read("clip.wav", dtype="float32")   # 16 kHz mono
assert sr == 16000
audio = (audio - audio.mean()) / (audio.std() + 1e-7)   # the model expects this

logp = sess.run(None, {"audio": audio[None, :]})[0][0]  # [frames, 105]

# Greedy CTC decode: collapse repeats, drop blanks (index 0).
ids = logp.argmax(-1)
prev, out = -1, []
for i in ids:
    if i != prev and i != 0:
        out.append(vocab[i])
    prev = i
print("".join(out))
```

For better accuracy, feed the per-frame log probabilities to a beam-search
decoder with a word-level language model — e.g.
[pyctcdecode](https://github.com/kensho-technologies/pyctcdecode) with a KenLM
model built from Sinhala text. That is worth roughly 13 WER points over greedy.

## Accuracy

Measured on a held-out test set that is disjoint by **speaker and by
sentence** — many published Sinhala numbers are measured on splits whose test
sentences also appear in training, which inflates them substantially.

| decoding | WER | CER |
|---|---|---|
| greedy (this model alone) | ~45% | ~11% |
| + word-level LM beam search | ~32% | ~7% |

For reference, on the same audio: stock Whisper-small scores 390% WER
(hallucination), and the unmodified base model 102.6% — it transcribes
Sinhala into Bengali script, which is the observation this model exists to fix.

## Limitations

- Trained on **read speech**. Spontaneous, colloquial Sinhala is harder:
  spoken word forms and word pairs are underrepresented in both the training
  audio and typical text corpora.
- No punctuation, no casing, no numerals-as-digits.
- Expects clean-ish 16 kHz mono input; it was built for close-talk phone
  microphones.
- The remaining errors are mostly real Sinhala words substituted in context —
  a language model helps, a dictionary alone does not.

## License and attribution

Released under the [Apache License 2.0](LICENSE).

- Base encoder: [facebook/omniASR-W2V-300M](https://huggingface.co/facebook/omniASR-W2V-300M), Apache-2.0, © Meta AI.
- Training data: [OpenSLR SLR52](https://openslr.org/52/) — "Large Sinhala ASR training data set",
  © Google, licensed CC BY-SA 4.0.
