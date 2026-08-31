"""
Web demo for the published sinhala-asr model.

Two decode lanes, shown side by side for the same recording:

  model alone     greedy CTC over just the model files (graph, weights, vocab)
  full pipeline   the same per-frame probabilities through beam search with
                  the word-level LM and lexicon from the release page, at the
                  decoder settings retuned on real dictation rather than read
                  speech, then through the word-repair pass that splits words
                  the search welded together

The greedy lane exists for comparison; the pipeline lane is the result. The
language model files are required -- without them the demo refuses to start,
because greedy-only output would be mistaken for the finished product.

Requires: flask, onnxruntime, soundfile, numpy, and ffmpeg on PATH.
The second lane also needs pyctcdecode and kenlm, plus a word-level ARPA
language model and unigram list under lm/ (too large for the repo --
download them from the release page, as described in the README).

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
    """The released model, decoded bare and through the full pipeline."""

    unfuser = None

    def __init__(self, model_dir: Path, lm_dir: Path) -> None:
        self.vocab: list[str] = json.loads((model_dir / "vocab.json").read_text("utf-8"))
        self.session = ort.InferenceSession(str(model_dir / "sinhala_asr_int8.onnx"))
        log.info("model loaded from %s", model_dir)
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
        log.info("language model loaded")

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
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", src.name,
                 "-ar", str(SAMPLE_RATE), "-ac", "1", dst.name],
                capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            raise ValueError("ffmpeg took over 30s to convert the recording")
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


PAGE_FILE = Path(__file__).resolve().parent / "demo.html"


def load_page() -> str:
    """The demo page, read from disk on each request.

    Kept in demo.html rather than a string literal here: it is HTML, CSS and
    JavaScript, and embedding it in Python cost it syntax highlighting and made
    every escape ambiguous. Re-read per request so the page can be edited
    without restarting the server -- this is a local demo, and the file is a few
    kilobytes.
    """
    try:
        return PAGE_FILE.read_text("utf-8")
    except OSError as exc:
        raise SystemExit(
            f"Cannot read the demo page at {PAGE_FILE}: {exc}\n"
            "It ships alongside demo.py -- re-clone if it has gone missing.")


def create_app(recogniser: Recogniser) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return load_page()

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
    parser.add_argument("--model-dir", type=Path, default=here / "model",
                        help="directory with sinhala_asr_int8.onnx, weights.bin, vocab.json")
    parser.add_argument("--lm-dir", type=Path, default=here / "lm",
                        help="directory with sinhala.arpa + unigrams.txt "
                             "(required; see the release page)")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    recogniser = Recogniser(args.model_dir, args.lm_dir)
    create_app(recogniser).run(host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
