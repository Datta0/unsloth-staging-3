"""Studio dictation of a downloaded mlx-audio checkpoint on Apple Silicon (unsloth#10272).

Runs the checked-out tree's own Studio backend: the HF cache inventory row, the Transformers
dictation sidecar and its spawned worker, against a real mlx-community Qwen3-ASR quant and speech
rendered by macOS `say`. Nothing is patched.

    python jobs/mlx_stt.py                 # mlx-community/Qwen3-ASR-0.6B-8bit
    python jobs/mlx_stt.py --model mlx-community/Qwen3-ASR-1.7B-8bit

Installs the Studio backend requirements and the Mac Studio pins (transformers 5.5.0 and the
tokenizers it accepts, mlx-vlm <= 0.7.4, which brings mlx-audio) on first run.

Checks: inventory lists the repo as ASR under library mlx-audio and not chattable; the sidecar sees
it as downloaded; a short and a > 30 s clip transcribe with the spoken words; unload frees it.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

DEFAULT_MODEL = "mlx-community/Qwen3-ASR-0.6B-8bit"
SHORT = "In the beginning God created the heaven and the earth."
LONG = (
    "And the earth was without form, and void; and darkness was upon the face of the deep. "
    "And the Spirit of God moved upon the face of the waters. And God said, Let there be light: "
    "and there was light. And God saw the light, that it was good: and God divided the light from "
    "the darkness. And God called the light Day, and the darkness he called Night. And the evening "
    "and the morning were the first day."
)
STUDIO_PINS = [
    "mlx==0.32.3",
    "mlx-metal==0.32.3",
    "mlx-lm==0.31.3",
    "mlx-vlm<=0.7.4",
    "av",
    "soundfile",
]
STUDIO_OVERRIDES = ["transformers==5.5.0", "tokenizers==0.22.2"]


def _words(text):
    return re.findall(r"[a-z]+", text.lower())


def _recall(expected, got):
    want, have = _words(expected), set(_words(got))
    return sum(w in have for w in want) / max(1, len(want))


def _say(
    text,
    out_wav,
    rate = 16000,
):
    aiff = out_wav.with_suffix(".aiff")
    subprocess.run(["say", "-o", str(aiff), text], check = True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", f"LEI16@{rate}", "-c", "1", str(aiff), str(out_wav)],
        check = True,
    )
    with wave.open(str(out_wav), "rb") as w:
        return w.getnframes() / w.getframerate()


def _pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check = True)


def main(argv = None):
    p = C.base_parser("mlx_stt", DEFAULT_MODEL, DEFAULT_MODEL, default_steps = 0)
    p.add_argument("--repo-root", default = ".", help = "checkout whose studio/backend runs")
    p.add_argument("--no-install", action = "store_true")
    a = C.resolve_args(p, argv)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    with C.JobRecorder(a, backend_hint = "studio-stt-mlx") as rec:
        if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            rec.skip(f"MLX needs Apple Silicon, this is {platform.system()}-{platform.machine()}")
        backend = (Path(a.repo_root) / "studio" / "backend").resolve()
        if not (backend / "core" / "inference" / "stt_sidecar.py").is_file():
            raise SystemExit(f"no Studio backend at {backend}")
        if not a.no_install:
            _pip("-r", str(backend / "requirements" / "studio.txt"))
            _pip(*STUDIO_PINS)
            _pip("--no-deps", *STUDIO_OVERRIDES)
        import importlib.metadata as md

        rec.data["summary"]["stack"] = {
            name: (md.version(name) if importlib_has(name) else None)
            for name in (
                "mlx",
                "mlx-lm",
                "mlx-vlm",
                "mlx-audio",
                "transformers",
                "tokenizers",
                "torch",
            )
        }

        hub = Path(tempfile.mkdtemp(prefix = "mlx_stt_hub_"))
        os.environ["HF_HUB_CACHE"] = str(hub)
        os.environ["UNSLOTH_STUDIO_HOME"] = tempfile.mkdtemp(prefix = "mlx_stt_home_")
        from huggingface_hub import snapshot_download

        t0 = time.perf_counter()
        snapshot = Path(snapshot_download(a.model, cache_dir = str(hub)))
        rec.data["summary"]["download_s"] = round(time.perf_counter() - t0, 1)
        os.environ["HF_HUB_OFFLINE"] = "1"

        sys.path.insert(0, str(backend))
        from core.inference import stt_mlx, stt_sidecar
        from hub.services.models import cache_inventory

        rec.check(
            "mlx_stt_available", stt_mlx.mlx_stt_available(), "Apple Silicon + mlx_audio importable"
        )
        rows = [
            r
            for r in cache_inventory._scan_cached_models(active_hub_cache = hub)
            if r["repo_id"].lower() == a.model.lower()
        ]
        row = rows[0] if rows else {}
        rec.check(
            "inventory_row_is_asr",
            row.get("task") == "automatic-speech-recognition"
            and row.get("library_name") == "mlx-audio",
            {k: row.get(k) for k in ("task", "library_name", "pipeline_tag")},
        )
        rec.check(
            "inventory_row_cannot_chat",
            (row.get("capabilities") or {}).get("can_chat") is False,
            row.get("capabilities"),
        )
        rec.check("sidecar_sees_download", stt_sidecar.is_model_downloaded(a.model), str(snapshot))

        audio_dir = Path(tempfile.mkdtemp(prefix = "mlx_stt_audio_"))
        clips = {
            "short": (SHORT, _say(SHORT, audio_dir / "short.wav")),
            "long": (LONG, _say(LONG, audio_dir / "long.wav")),
        }
        rec.data["summary"]["clip_seconds"] = {k: round(v[1], 1) for k, v in clips.items()}
        sidecar = stt_sidecar.WhisperSttSidecar(keep_alive_seconds = 0)
        try:
            for name, (text, seconds) in clips.items():
                t0 = time.perf_counter()
                result = sidecar.transcribe((audio_dir / f"{name}.wav").read_bytes(), model = a.model)
                took = time.perf_counter() - t0
                recall = _recall(text, result.get("text") or "")
                rec.step(
                    step = len(rec.data["steps"]) + 1,
                    clip = name,
                    audio_s = round(seconds, 1),
                    wall_s = round(took, 2),
                    text = result.get("text"),
                    word_recall = round(recall, 3),
                    device = sidecar.device,
                )
                rec.check(
                    f"transcribes_{name}_clip",
                    recall >= 0.8,
                    f"recall {recall:.2f} over {seconds:.1f}s: {result.get('text')!r}",
                )
            rec.check(
                "model_resident_on_engine",
                sidecar.loaded_model == a.model,
                f"loaded={sidecar.loaded_model} device={sidecar.device}",
            )
        finally:
            sidecar.unload()
        rec.check("unload_releases_model", sidecar.loaded_model is None, sidecar.loaded_model)


def importlib_has(name):
    import importlib.metadata as md
    try:
        md.version(name)
        return True
    except md.PackageNotFoundError:
        return False


if __name__ == "__main__":
    main()
