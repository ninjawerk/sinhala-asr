"""
Web demo for the published sinhala-asr model.

Two decode lanes, shown side by side for the same recording:

  released model alone   greedy CTC over the exact three files a stranger gets
                         from the GitHub release (graph, weights, vocab)
  full research pipeline the same per-frame probabilities through beam search
                         with a word-level LM and lexicon, at the decoder
                         settings retuned on real dictation rather than read
                         speech, then through the word-repair pass that splits
                         words the search welded together

The greedy lane exists for comparison; the pipeline lane is the result. The
language model files are required -- without them the demo refuses to start,
because greedy-only output would be mistaken for the finished product.

Requires: flask, onnxruntime, soundfile, numpy, and ffmpeg on PATH.
The second lane also needs pyctcdecode and kenlm, plus a word-level ARPA
language model and unigram list under lm/ (not included in this repo --
build one from Sinhala text with kenlm's lmplz, or any ARPA-producing tool).

    python demo.py
    open http://localhost:7861
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from flask import Flask, jsonify, request

log = logging.getLogger("oss-demo")

SAMPLE_RATE = 16_000
MIN_SAMPLES = SAMPLE_RATE // 5          # refuse clips under 200 ms
MAX_SECONDS = 60                        # and over a minute
BEAM_WIDTH = 128

# Retuned against real phone dictation, not read-speech corpus audio -- the
# corpus-tuned values (0.5 / 1.5) measured about 3.5 WER points worse on live
# speech. Using anything else here re-introduces a mistake the research fixed.
LM_ALPHA = 0.8
LM_BETA = 3.0


@dataclass
class Result:
    greedy: str
    beamed: str | None
    seconds: float
    decode_ms: int


class Recogniser:
    """The released model, plus the full research decode lane for comparison."""

    unfuser = None

    def __init__(self, repo: Path, lm_dir: Path | None) -> None:
        self.vocab: list[str] = json.loads((repo / "vocab.json").read_text("utf-8"))
        self.session = ort.InferenceSession(str(repo / "sinhala_asr_int8.onnx"))
        log.info("model loaded from %s", repo)
        self.lm_decoder = self._load_lm(lm_dir)

    def _load_lm(self, lm_dir: Path) -> object:
        arpa = lm_dir / "sinhala.arpa"
        unigrams = lm_dir / "unigrams.txt"
        if not (arpa.exists() and unigrams.exists()):
            # Refuse to run rather than degrade: greedy-only output is the
            # model at its roughest, and a demo that silently serves it would
            # be mistaken for the finished product.
            raise SystemExit(
                f"Language model files not found in {lm_dir}/.\n"
                "The demo needs them -- download from the release page:\n"
                "  mkdir -p lm\n"
                "  gunzip -c sinhala.arpa.gz  > lm/sinhala.arpa\n"
                "  gunzip -c unigrams.txt.gz > lm/unigrams.txt")
        from pyctcdecode import build_ctcdecoder

        labels = [""] + self.vocab[1:]
        charset = set(labels)
        words = [w.strip() for w in unigrams.read_text("utf-8").splitlines()]
        words = [w for w in words if w and all(c in charset for c in w)]
        decoder = build_ctcdecoder(labels, str(arpa), words[:400_000],
                                   alpha=LM_ALPHA, beta=LM_BETA)
        log.info("language model loaded (demo only, not in the release)")

        # The word-repair pass from the research: spoken Sinhala welds words
        # together, and the n-gram arithmetic can prefer the welded form. This
        # splits a non-word back into real words where the lexicon supports it.
        import kenlm
        from unfuse import WordRepair
        lexicon = {w for w in words}
        self.unfuser = WordRepair(lexicon, kenlm.Model(str(arpa)))
        log.info("word-repair pass loaded")
        return decoder

    def transcribe(self, wav: np.ndarray) -> Result:
        normalised = (wav - wav.mean()) / (wav.std() + 1e-7)
        started = time.time()
        logp = self.session.run(
            None, {"audio": normalised[None, :].astype(np.float32)})[0][0]

        greedy = self._greedy(logp)
        beamed = None
        if self.lm_decoder is not None:
            beamed = self.lm_decoder.decode(logp.astype(np.float32),
                                            beam_width=BEAM_WIDTH)
            beamed = self.unfuser.fix(beamed)
        return Result(greedy, beamed, len(wav) / SAMPLE_RATE,
                      int((time.time() - started) * 1000))

    def _greedy(self, logp: np.ndarray) -> str:
        """Collapse repeats, drop blanks (index 0) -- as in the README."""
        out: list[str] = []
        previous = -1
        for i in logp.argmax(-1):
            if i != previous and i != 0:
                out.append(self.vocab[i])
            previous = i
        return "".join(out)


def decode_upload(blob) -> np.ndarray:
    """Browser audio (webm/ogg/anything ffmpeg reads) -> 16 kHz mono float32."""
    with tempfile.NamedTemporaryFile(suffix=".webm") as src, \
         tempfile.NamedTemporaryFile(suffix=".wav") as dst:
        blob.save(src.name)
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", src.name,
             "-ar", str(SAMPLE_RATE), "-ac", "1", dst.name],
            capture_output=True, timeout=30)
        if proc.returncode != 0:
            raise ValueError("ffmpeg could not read the recording")
        wav, rate = sf.read(dst.name, dtype="float32")
    if rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz, got {rate}")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if len(wav) < MIN_SAMPLES:
        raise ValueError("recording too short -- hold the button and speak")
    if len(wav) > MAX_SECONDS * SAMPLE_RATE:
        raise ValueError(f"recording over {MAX_SECONDS}s")
    return wav


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>sinhala-asr demo</title>
<style>
body{background:#0e0f13;color:#eef0f4;font-family:-apple-system,Helvetica,Arial,sans-serif;
     max-width:640px;margin:0 auto;padding:40px 20px}
h1{font-size:1.5rem}h1 span{color:#e0873a}
p.sub{color:#9aa2b1;font-size:.95rem}
button{background:#e0873a;color:#1a0f05;border:0;border-radius:10px;padding:14px 22px;
       font-size:1rem;font-weight:700;cursor:pointer}
.box{background:#15171d;border:1px solid #242833;border-radius:10px;padding:18px;
     min-height:64px;font-size:1.4rem;line-height:1.6}
.lbl{color:#9aa2b1;font-size:.78rem;letter-spacing:.06em;text-transform:uppercase;
     margin:22px 0 6px}
.lbl .note{text-transform:none;letter-spacing:0;color:#6b7280}
#meta{color:#6b7280;font-family:monospace;font-size:.8rem;margin-top:12px}
#err{color:#e58060;font-size:.9rem;margin-top:12px}
a{color:#f0a257}
</style></head><body>
<h1>sinhala-asr <span>&middot; open-source model demo</span></h1>
<p class="sub">The exact files from
<a href="https://github.com/ninjawerk/sinhala-asr">github.com/ninjawerk/sinhala-asr</a>,
decoded two ways so the language model's contribution is visible.</p>

<p><button id="rec">&#127908; Record</button></p>

<div class="lbl">released model alone (greedy)</div>
<div class="box" id="greedy">&hellip;</div>

<div class="lbl">full research pipeline
  <span class="note">(+ language model, live-tuned decoder, word repair &mdash; not in the release)</span></div>
<div class="box" id="beamed">&hellip;</div>

<div id="meta"></div>
<div id="err"></div>

<script>
const el = id => document.getElementById(id);
let recorder = null;

el('rec').onclick = async () => {
  if (recorder && recorder.state === 'recording') { recorder.stop(); return; }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    el('err').textContent = 'Microphone unavailable: ' + e.message;
    return;
  }
  const chunks = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = e => chunks.push(e.data);
  recorder.onstop = async () => {
    el('rec').textContent = '\\u{1F3A4} Record';
    stream.getTracks().forEach(t => t.stop());
    el('greedy').innerHTML = '&hellip;';
    el('beamed').innerHTML = '&hellip;';
    el('err').textContent = '';
    el('meta').textContent = 'transcribing\\u2026';
    const body = new FormData();
    body.append('audio', new Blob(chunks, { type: recorder.mimeType }));
    try {
      const resp = await fetch('/transcribe', { method: 'POST', body });
      const j = await resp.json();
      if (!resp.ok) throw new Error(j.error || resp.statusText);
      el('greedy').textContent = j.greedy || '(nothing)';
      el('beamed').textContent = j.beamed || '(nothing)';
      el('meta').textContent =
        j.seconds.toFixed(2) + 's audio \\u00b7 decoded in ' + j.decode_ms + ' ms';
    } catch (e) {
      el('meta').textContent = '';
      el('err').textContent = e.message;
    }
  };
  recorder.start();
  el('rec').textContent = '\\u23F9 Stop';
};
</script></body></html>"""


def create_app(recogniser: Recogniser) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return PAGE

    @app.post("/transcribe")
    def transcribe():
        if "audio" not in request.files:
            return jsonify(error="no audio field"), 400
        try:
            wav = decode_upload(request.files["audio"])
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        result = recogniser.transcribe(wav)
        log.info("%.2fs -> %r", result.seconds, result.greedy)
        return jsonify(greedy=result.greedy, beamed=result.beamed,
                       seconds=result.seconds, decode_ms=result.decode_ms)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--repo", type=Path, default=here,
                        help="directory with sinhala_asr_int8.onnx, weights.bin, vocab.json")
    parser.add_argument("--lm-dir", type=Path, default=here / "lm",
                        help="directory with sinhala.arpa + unigrams.txt "
                             "(required; see the release page)")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    recogniser = Recogniser(args.repo, args.lm_dir)
    create_app(recogniser).run(host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
