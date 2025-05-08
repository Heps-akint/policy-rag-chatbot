"""
Smoke test for the local 4-bit Mistral-7B-GPTQ checkpoint.

Usage
-----
python smoke_mistral.py [MODEL_DIR] [--bench]

* MODEL_DIR — path to a folder that contains `model.safetensors` and
  `config.json`. Defaults to `models/mistral/gptq` relative to the project
  root.
* --bench — generate 100 tokens instead of 3 so you get a realistic
  throughput number.

The script logs the cold-load latency and prints the generated text via the
standard logging module so it passes Ruff's T201 rule (no raw print).
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import time

from gptqmodel import GPTQModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Default path: project_root/models/mistral/gptq
DEFAULT_MODEL_DIR = (
    pathlib.Path(__file__).parent.parent / "models" / "mistral" / "gptq"
).resolve()


def load_model(model_path: pathlib.Path) -> GPTQModel:
    """Return a GPTQ-quantised Mistral model loaded onto GPU (or CPU)."""
    t0 = time.time()
    model = GPTQModel.load(
        model_path,
        device="cuda:0",  # use "cpu" if no GPU
        exllama_config={"version": 2},  # fastest CUDA kernels
    )
    logger.info("Loaded model in %.1fs", time.time() - t0)
    return model


def main() -> None:
    """Parse CLI flags, load the model, and run a short generation test."""
    parser = argparse.ArgumentParser(description="Mistral GPTQ smoke-test")
    parser.add_argument(
        "model_dir",
        nargs="?",
        default=DEFAULT_MODEL_DIR,
        type=pathlib.Path,
        help="Path to GPTQ model directory",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Generate 100 tokens to benchmark throughput",
    )
    args = parser.parse_args()

    model = load_model(args.model_dir)

    prompt = "Hello"
    max_new = 100 if args.bench else 3
    tokens = model.generate(prompt, max_new_tokens=max_new)[0]
    logger.info("%s", model.tokenizer.decode(tokens))


if __name__ == "__main__":
    main()
