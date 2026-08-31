# sinhala-asr

Offline Sinhala speech recognition. A 300M-parameter model, shrunk to run on a
phone — it transcribes 16 kHz Sinhala speech several times faster than real
time, entirely on-device.

## Quick start

1. Clone this repo.
2. From the [release page](../../releases), download:
   - `weights.bin` — put it next to `sinhala_asr_int8.onnx`
   - `sinhala.arpa.gz` and `unigrams.txt.gz` — the language model:

   ```bash
   mkdir -p lm
   gunzip -c sinhala.arpa.gz  > lm/sinhala.arpa
   gunzip -c unigrams.txt.gz > lm/unigrams.txt
   ```

3. Install and run the demo:

   ```bash
   pip install -r requirements.txt      # also needs ffmpeg installed
   python demo.py                       # open http://localhost:7861
   ```

Press record, speak Sinhala, and see the result.

## Using it in your own code

The full pipeline — model, language model, word repair:

```python
import json
import numpy as np
import onnxruntime as ort
import soundfile as sf
import kenlm
from pyctcdecode import build_ctcdecoder
from unfuse import WordRepair

vocab = json.load(open("vocab.json", encoding="utf-8"))
sess = ort.InferenceSession("sinhala_asr_int8.onnx")

audio, sr = sf.read("clip.wav", dtype="float32")   # 16 kHz mono
assert sr == 16000
audio = (audio - audio.mean()) / (audio.std() + 1e-7)   # required
logp = sess.run(None, {"audio": audio[None, :]})[0][0]

labels = [""] + vocab[1:]
charset = set(labels)
words = [w.strip() for w in open("lm/unigrams.txt", encoding="utf-8") if w.strip()]
words = [w for w in words if all(c in charset for c in w)]
decoder = build_ctcdecoder(labels, "lm/sinhala.arpa", words, alpha=0.8, beta=3.0)
text = decoder.decode(logp.astype(np.float32), beam_width=128)

repair = WordRepair(set(words), kenlm.Model("lm/sinhala.arpa"))
print(repair.fix(text))
```

If you only want the raw model — for example to plug in your own decoder —
greedy decoding is just: collapse repeated symbols, drop index 0 (the blank),
join what's left. Expect much rougher output; the language model is where a
third of the accuracy comes from.

## How it works

- The **encoder** is Meta's open
  [omniASR-W2V-300M](https://huggingface.co/facebook/omniASR-W2V-300M),
  used unchanged. It turns audio into rich sound features.
- On top sits a small **Sinhala output layer**, trained on ~185k clips of
  transcribed Sinhala speech ([OpenSLR SLR52](https://openslr.org/52/)).
  This is the only part that was trained.
- Both are fused into one ONNX graph and quantised to **int8** — about a third
  of the original size, with output verified character-for-character
  identical to full precision.

The model outputs, 50 times per second of audio, a probability for each of
105 symbols — Sinhala, plus a handful of Latin letters and digits that occur
in the training transcripts (loanwords, brand names). Turning those into
text is the decoder's job — the simple greedy loop above, or a smarter
decoder with a language model.

## Files

| file | size | where |
|---|---|---|
| `sinhala_asr_int8.onnx` | 0.8 MB | this repo |
| `vocab.json` | 1 KB | this repo |
| `weights.bin` | 342 MB | [release](../../releases) |
| `sinhala.arpa.gz` | 120 MB | [release](../../releases) — word-level language model |
| `unigrams.txt.gz` | 2.5 MB | [release](../../releases) — 400k-word list |

## Accuracy

Measured on speakers **and** sentences the model never saw in training.
(Many published Sinhala numbers use test sentences that also appear in the
training data, which makes them look better than they are.)

| decoding | word error rate |
|---|---|
| greedy (model alone) | ~45% |
| + language model beam search | ~32% |

The decoder settings that work best on real dictation are
`alpha=0.8, beta=3.0` — see `demo.py`. For scale: stock Whisper-small scores
390% on the same audio, and the unmodified base model 102.6% — it writes
Sinhala in Bengali script, which is the problem this model exists to fix.

## Demo

`demo.py` is a small local web page: press record, speak Sinhala, and see the
same recording decoded two ways — the model alone, and the full pipeline with
the language model and word repair.

It will not start without the `lm/` files from Quick start: the model alone
is deliberately rough, and a demo of it would give the wrong impression of
what this project does.

`unfuse.py` is the word-repair step. Spoken Sinhala runs words together, and
the decoder sometimes writes several words as one long non-word. This pass
checks whether such a word splits cleanly into real words, and fixes it when
it does.

The language model was built from public Sinhala web text (news, blogs,
subtitles) plus the SLR52 transcripts. It contains word statistics, not the
original documents.

## Keeping memory low

On a phone this runs in about **232 MB of resident memory** (measured on a
Galaxy S24 Ultra). You don't get that by default — it comes from four habits:

**1. Don't load the big files — map them.** Keep `weights.bin` on disk next
to the graph and let onnxruntime find it there. The OS pages it in as needed
and can reclaim the memory under pressure. Reading it into a byte array
yourself makes that impossible.

**2. Turn off onnxruntime's memory arena.** By default it grabs memory
greedily and never gives it back:

```python
opts = ort.SessionOptions()
opts.enable_cpu_mem_arena = False
opts.intra_op_num_threads = 4
sess = ort.InferenceSession("sinhala_asr_int8.onnx", opts)
```

**3. Cut long recordings into windows.** Memory grows with audio length.
Transcribe overlapping windows of a few seconds each, drop the frames near
the seams, and normalise over the whole recording — memory then depends on
the window size, not the recording length.

**4. Convert the language model to kenlm's binary format.** The text ARPA
loads fully into memory; the binary format is memory-mapped instead:

```bash
build_binary trie lm/sinhala.arpa lm/sinhala.klm
```

None of these change the output by a single character. Don't take that on
trust — diff the transcripts before and after, which is how everything in
this README was checked.

## Limitations

- Trained on **read speech** — people reading sentences aloud. Natural, fast,
  colloquial speech is harder for it.
- No punctuation. Latin letters and digits are in the vocabulary (they appear
  in the training transcripts), but the model emits them too rarely to rely on.
- Wants clean 16 kHz mono audio, close to the microphone.
- Its remaining mistakes are mostly real Sinhala words in the wrong place — a
  language model helps with that; a dictionary alone cannot.

## License and attribution

[Apache 2.0](LICENSE). Free for any use, including commercial.

One thing the license asks of you: if you ship this model in a product, keep
the [NOTICE](NOTICE) file's contents with it (Apache 2.0, section 4d). In
practice that means a line crediting this project somewhere in your app or
its documentation:

> Sinhala speech recognition: sinhala-asr by Deshan Alahakoon
> — github.com/ninjawerk/sinhala-asr

This project itself builds on:

- Base encoder: [facebook/omniASR-W2V-300M](https://huggingface.co/facebook/omniASR-W2V-300M), Apache-2.0, © Meta AI.
- Training data: [OpenSLR SLR52](https://openslr.org/52/), © Google, CC BY-SA 4.0.
