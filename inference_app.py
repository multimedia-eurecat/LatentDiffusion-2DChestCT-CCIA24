#!/usr/bin/env python3
import os
import sys
import glob
import uuid
import time
import shutil
from pathlib import Path
from threading import Lock, Thread
from contextlib import redirect_stdout, redirect_stderr
from math import ceil

import torch
from tqdm import tqdm
from flask import (
    Flask,
    request,
    send_file,
    url_for,
    render_template_string,
    abort,
    session,
    redirect,
)

# Reuse the repo's existing utilities.
sys.path.insert(1, "./src")
from utils_lidc import *
from custom_unet_cond import *

from generate_N_images import save_pipeline_output_to_disk


# =========================
# MANUALLY SET THESE PATHS
# =========================
UNCONDITIONAL_CKPT_DIR = "data/ckpts/2024-06-12_18-01-52_FINAL_CCIA24_unconditional_latent_BS8"
MASK_CONDITIONED_CKPT_DIR = "data/ckpts/2024-07-12_10-03-18_FINAL_CCIA24_model2_crossattention_locmask_maskv2"
MASK_ATTR_CONDITIONED_CKPT_DIR = "data/ckpts/2024-07-12_19-57-59_FINAL_CCIA24_model3_crossattention_locmask_noduleattributes_maskv2"

MASKS_DIR = "data/train_data/masks_6mm_512x512_sq"
# MASKS_DIR = None

# Set this as desired. Examples: "cpu", "cuda", "cuda:0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HOST = "0.0.0.0"
PORT = 7860
WEB_OUTPUTS_DIR = "web_outputs"


MODEL_CONFIGS = {
    "unconditional": {
        "label": "Unconditional",
        "ckpt_dir": UNCONDITIONAL_CKPT_DIR,
        "masks_dir": None,
    },
    "mask": {
        "label": "Conditioned to nodule masks",
        "ckpt_dir": MASK_CONDITIONED_CKPT_DIR,
        "masks_dir": MASKS_DIR,
    },
    "mask_attr": {
        "label": "Conditioned to nodule masks + nodule attributes",
        "ckpt_dir": MASK_ATTR_CONDITIONED_CKPT_DIR,
        "masks_dir": MASKS_DIR,
    },
}

MODEL_CHOICES = [
    ("unconditional", "Unconditional"),
    ("mask", "Conditioned to nodule masks"),
    ("mask_attr", "Conditioned to nodule masks + nodule attributes"),
]

PARAM_LIMITS = {
    "n_images": {"min": 1, "max": 500},
    "batch_size": {"min": 1, "max": 10},
    "inference_steps": {"min": 1, "max": 1000},
}


def human_readable_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024.0


def validate_int_param(name: str, value: int) -> int:
    limits = PARAM_LIMITS[name]
    value = int(value)
    if value < limits["min"] or value > limits["max"]:
        raise ValueError(f"{name} must be between {limits['min']} and {limits['max']}.")
    return value


def zip_output_dir(out_dir: str) -> str:
    parent_dir = os.path.dirname(out_dir)
    base_name = os.path.join(parent_dir, Path(out_dir).name + "_synthetic_data")
    return shutil.make_archive(base_name, "zip", root_dir=out_dir)


def first_generated_image(out_dir: str, use_overlay: bool = False) -> str:
    preview_dir = "overlay" if use_overlay else "images"
    image_paths = sorted(glob.glob(os.path.join(out_dir, preview_dir, "*.png")))
    if not image_paths:
        raise RuntimeError(f"No preview images were generated in '{preview_dir}'.")
    return image_paths[0]


def generate_images_web(
    pipeline,
    out_dir: str,
    n_images: int,
    batch_size: int = 16,
    inference_steps: int | None = None,
    logger=None,
):
    def log(msg: str):
        if logger is not None:
            logger.write(msg.rstrip() + "\n")

    generator = torch.Generator(device=DEVICE)
    save_masks = pipeline.mask_generator is not None

    if inference_steps is None:
        inference_steps = pipeline.scheduler.config.num_train_timesteps

    im_size = pipeline.vae.sample_size
    full_batches = n_images // batch_size
    remainder = n_images % batch_size

    log(f"Device: {DEVICE}")
    log(f"Output folder: {out_dir}")
    log(f"Number of images: {n_images}")
    log(f"Batch size: {batch_size}")
    log(f"Inference steps: {inference_steps}")
    log(f"Conditional mode: {'yes' if save_masks else 'no'}")
    log(f"Full batches: {full_batches}")
    log(f"Remainder batch size: {remainder}")

    for batch_idx in range(full_batches):
        log(f"Running batch {batch_idx + 1}/{full_batches}")
        output = pipeline(
            height=im_size,
            width=im_size,
            generator=generator,
            batch_size=batch_size,
            num_inference_steps=inference_steps,
            output_type="numpy",
            return_dict=False,
        )
        save_pipeline_output_to_disk(
            output=output,
            batch_idx=batch_idx,
            batch_size=batch_size,
            out_dir=out_dir,
            save_masks=save_masks,
        )
        log(f"Saved batch {batch_idx + 1}/{full_batches}")

    if remainder > 0:
        batch_idx = full_batches
        log("Running final partial batch")
        output = pipeline(
            height=im_size,
            width=im_size,
            generator=generator,
            batch_size=remainder,
            num_inference_steps=inference_steps,
            output_type="numpy",
            return_dict=False,
        )
        save_pipeline_output_to_disk(
            output=output,
            batch_idx=batch_idx,
            batch_size=batch_size,
            out_dir=out_dir,
            save_masks=save_masks,
        )
        log("Saved final partial batch")

    log("Generation finished")


class GeneratorApp:
    def __init__(self):
        self.pipelines = {}
        self.lock = Lock()

    def get_model_config(self, model_key: str):
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_key}")
        return MODEL_CONFIGS[model_key]

    def load_pipeline_for_model(self, model_key: str):
        if model_key in self.pipelines:
            return self.pipelines[model_key]

        cfg = self.get_model_config(model_key)
        ckpt_dir = cfg["ckpt_dir"]
        masks_dir = cfg["masks_dir"]

        if not ckpt_dir or not os.path.exists(ckpt_dir):
            raise FileNotFoundError(
                f"Checkpoint directory not found for model '{model_key}': {ckpt_dir}"
            )

        pipeline = load_pipeline(ckpt_dir, masks_dir, device=DEVICE)
        self.pipelines[model_key] = pipeline
        return pipeline

    def create_run(self, model_key: str, n_images: int, batch_size: int, inference_steps: int):
        n_images = validate_int_param("n_images", n_images)
        batch_size = validate_int_param("batch_size", batch_size)
        inference_steps = validate_int_param("inference_steps", inference_steps)

        run_id = f"{model_key}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        out_dir = os.path.join(WEB_OUTPUTS_DIR, run_id)
        os.makedirs(out_dir, exist_ok=True)

        with RUNS_LOCK:
            RUNS[run_id] = {
                "status": "running",
                "logs": "",
                "error": None,
                "result": None,
                "updated_at": time.time(),
                "form_values": {
                    "model_key": model_key,
                    "n_images": n_images,
                    "batch_size": batch_size,
                    "inference_steps": inference_steps,
                },
            }

        thread = Thread(
            target=self._run_generation_task,
            args=(run_id, model_key, n_images, batch_size, inference_steps, out_dir),
            daemon=True,
        )
        thread.start()
        return run_id

    def _run_generation_task(
        self,
        run_id: str,
        model_key: str,
        n_images: int,
        batch_size: int,
        inference_steps: int,
        out_dir: str,
    ):
        logger = RunLogger(run_id)

        try:
            cfg = self.get_model_config(model_key)
            append_run_log(run_id, f"Selected model: {cfg['label']}")
            append_run_log(run_id, "Loading pipeline...")

            with self.lock:
                pipeline = self.load_pipeline_for_model(model_key)
                append_run_log(run_id, "Pipeline loaded")
                append_run_log(run_id, "Starting image generation...")

                with redirect_stdout(logger), redirect_stderr(logger):
                    generate_images_web(
                        pipeline=pipeline,
                        out_dir=out_dir,
                        n_images=n_images,
                        batch_size=batch_size,
                        inference_steps=inference_steps,
                        logger=logger,
                    )

            use_overlay = model_key in {"mask", "mask_attr"}
            preview_subdir = "overlay" if use_overlay else "images"
            preview_path = first_generated_image(out_dir, use_overlay=use_overlay)
            zip_path = zip_output_dir(out_dir)
            zip_size_bytes = os.path.getsize(zip_path)

            result = {
                "model_label": cfg["label"],
                "out_dir": out_dir,
                "preview_path": preview_path,
                "preview_subdir": preview_subdir,
                "zip_path": zip_path,
                "zip_size_bytes": zip_size_bytes,
                "zip_size_human": human_readable_size(zip_size_bytes),
                "run_id": Path(out_dir).name,
                "n_images": n_images,
            }

            with RUNS_LOCK:
                RUNS[run_id]["status"] = "done"
                RUNS[run_id]["result"] = result
                RUNS[run_id]["updated_at"] = time.time()

            append_run_log(
                run_id,
                f"Zip created: {Path(zip_path).name} ({human_readable_size(zip_size_bytes)})",
            )
            append_run_log(run_id, "Run completed successfully")

        except Exception as e:
            with RUNS_LOCK:
                if run_id in RUNS:
                    RUNS[run_id]["status"] = "error"
                    RUNS[run_id]["error"] = str(e)
                    RUNS[run_id]["updated_at"] = time.time()
            append_run_log(run_id, f"ERROR: {e}")


generator_app = GeneratorApp()
app = Flask(__name__)

# Simple password auth for public deployments.
# If PASSWORD is empty/unset, auth is disabled.
AUTH_PASSWORD = os.getenv("PASSWORD", "")
AUTH_ENABLED = bool(AUTH_PASSWORD)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(32))

RUNS = {}
RUNS_LOCK = Lock()


LOGIN_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Login</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 32px; max-width: 420px; }
    h1 { margin-bottom: 16px; }
    .panel { border: 1px solid #ddd; border-radius: 10px; padding: 18px; }
    label { display: block; margin-top: 8px; font-weight: 600; }
    input[type=password] { width: 100%; padding: 10px; margin-top: 6px; box-sizing: border-box; }
    button { margin-top: 14px; padding: 10px 16px; border: 0; border-radius: 8px; background: #0b5fff; color: white; cursor: pointer; }
    .error { background: #ffe8e8; color: #8a1f1f; border: 1px solid #f1b5b5; padding: 10px; border-radius: 8px; margin-bottom: 12px; }
    .muted { color: #666; font-size: 13px; }
  </style>
</head>
<body>
  <h1>Protected UI</h1>
  <div class="panel">
    {% if error %}
      <div class="error">{{ error }}</div>
    {% endif %}
    <form method="post">
      <label for="password">Password</label>
      <input type="password" id="password" name="password" autofocus required>
      <button type="submit">Login</button>
    </form>
    <div class="muted" style="margin-top:10px;">Access is controlled by env var: PASSWORD</div>
  </div>
</body>
</html>
"""


@app.before_request
def require_password_auth():
    if not AUTH_ENABLED:
        return None

    allowed_paths = {"/login"}
    if request.path in allowed_paths:
        return None

    if session.get("authenticated") is True:
        return None

    return redirect(url_for("login", next=request.path))


@app.route("/login", methods=["GET", "POST"])
def login():
    if not AUTH_ENABLED:
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        provided = request.form.get("password", "")
        if provided == AUTH_PASSWORD:
            session["authenticated"] = True
            next_path = request.args.get("next") or "/"
            if not next_path.startswith("/"):
                next_path = "/"
            return redirect(next_path)
        error = "Invalid password"

    return render_template_string(LOGIN_TEMPLATE, error=error)


@app.route("/logout", methods=["GET"])
def logout():
    session.pop("authenticated", None)
    return redirect(url_for("login"))


class RunLogger:
    def __init__(self, run_id: str):
        self.run_id = run_id

    def write(self, text):
        if not text:
            return
        with RUNS_LOCK:
            if self.run_id in RUNS:
                RUNS[self.run_id]["logs"] += text
                RUNS[self.run_id]["updated_at"] = time.time()

    def flush(self):
        pass


def append_run_log(run_id: str, text: str):
    if not text:
        return
    with RUNS_LOCK:
        if run_id in RUNS:
            RUNS[run_id]["logs"] += text + "\n"
            RUNS[run_id]["updated_at"] = time.time()


HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Synthetic Chest CT Generator</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 32px; max-width: 1200px; }
    h1 { margin-bottom: 24px; }
    .grid { display: grid; grid-template-columns: 320px 1fr; gap: 28px; align-items: start; }
    .panel { border: 1px solid #ddd; border-radius: 10px; padding: 18px; margin-bottom: 20px; }
    label { display: block; margin-top: 12px; font-weight: 600; }
    select, input[type=number] { width: 100%; padding: 10px; margin-top: 6px; box-sizing: border-box; }
    button, .download-btn {
      margin-top: 18px; padding: 12px 18px; border: 0; border-radius: 8px;
      background: #0b5fff; color: white; cursor: pointer; text-decoration: none; display: inline-block;
    }
    .download-btn { background: #198754; }
    .thumb { max-width: 100%; border-radius: 8px; border: 1px solid #ddd; }
    .row { display: flex; gap: 18px; align-items: center; }
    .error { background: #ffe8e8; color: #8a1f1f; border: 1px solid #f1b5b5; padding: 12px; border-radius: 8px; margin-bottom: 18px; }
    .meta { white-space: pre-line; color: #333; }
    .muted { color: #666; font-size: 14px; }
    .status { margin-top: 12px; font-weight: 600; }
    .log-box {
        height: 420px;
        overflow: auto;
        background: #111;
        color: #eee;
        padding: 12px;
        border-radius: 8px;
        white-space: pre;
        font-family: monospace;
        font-size: 13px;
        line-height: 1.4;
        border: 1px solid #333;
    }
  </style>
</head>
<body>
  <h1>Synthetic Chest CT Generator</h1>

  {% if error %}
    <div class="error">{{ error }}</div>
  {% endif %}

  <div class="grid">
    <div>
      <div class="panel">
        <form method="post" action="/generate">
          <label for="model_key">Model checkpoint</label>
          <select name="model_key" id="model_key">
            {% for value, label in model_choices %}
              <option value="{{ value }}" {% if form_values.model_key == value %}selected{% endif %}>{{ label }}</option>
            {% endfor %}
          </select>

          <label for="n_images">Number of images</label>
          <input
            type="number"
            id="n_images"
            name="n_images"
            min="{{ param_limits.n_images.min }}"
            max="{{ param_limits.n_images.max }}"
            step="1"
            value="{{ form_values.n_images }}"
          >
          <div class="muted">
            Number of images to generate (min: {{ param_limits.n_images.min }}, max: {{ param_limits.n_images.max }})
          </div>

          <label for="batch_size">Batch size</label>
          <input
            type="number"
            id="batch_size"
            name="batch_size"
            min="{{ param_limits.batch_size.min }}"
            max="{{ param_limits.batch_size.max }}"
            step="1"
            value="{{ form_values.batch_size }}"
          >
          <div class="muted">
            Number of images that will be generated simultaneously (min: {{ param_limits.batch_size.min }}, max: {{ param_limits.batch_size.max }})
          </div>

          <label for="inference_steps">Inference steps</label>
          <input
            type="number"
            id="inference_steps"
            name="inference_steps"
            min="{{ param_limits.inference_steps.min }}"
            max="{{ param_limits.inference_steps.max }}"
            step="1"
            value="{{ form_values.inference_steps }}"
          >
          <div class="muted">
            Number of generative steps, higher values provide higher quality outputs but require longer computation time (min: {{ param_limits.inference_steps.min }}, max: {{ param_limits.inference_steps.max }})
          </div>

          <button type="submit">Generate synthetic data</button>
        </form>
      </div>

    </div>

    <div>
      <div class="panel">
        {% if result %}
          <div class="row">
            <div style="min-width: 320px;">
              <img class="thumb" src="{{ result.preview_url }}" alt="Thumbnail">
            </div>
            <div>
              <a class="download-btn" href="{{ result.download_url }}">
                Download synthetic data ({{ result.zip_size_human }})
              </a>
              <div class="meta" style="margin-top: 16px;">
Model: {{ result.model_label }}
Generated {{ result.n_images }} image(s)
Preview: {{ result.preview_name }}
Zip: {{ result.zip_name }}
Zip size: {{ result.zip_size_human }}
Output folder: {{ result.out_dir }}
              </div>
            </div>
          </div>
        {% else %}
          <div class="muted">No generation result yet.</div>
        {% endif %}
      </div>
    </div>
  </div>

  
  {% if run_id %}
  <div class="panel">
    <div class="status">Status: <span id="run-status">{{ run_status }}</span></div>
    <div class="muted" style="margin-top: 8px;">Command line output</div>
    <pre id="log-box" class="log-box">{{ logs }}</pre>
  </div>
  {% endif %}


{% if run_id and run_status == "running" %}
<script>
  async function pollRun(runId) {
    const logBox = document.getElementById("log-box");
    const statusEl = document.getElementById("run-status");

    while (true) {
      const res = await fetch(`/run_status/${runId}`);
      const data = await res.json();

      if (statusEl) {
        statusEl.textContent = data.status;
      }
      if (logBox) {
        logBox.textContent = data.logs || "";
        logBox.scrollTop = logBox.scrollHeight;
      }

      if (data.status === "done") {
        window.location.href = `/run/${runId}`;
        break;
      }

      if (data.status === "error") {
        if (statusEl) {
          statusEl.textContent = "error";
        }
        break;
      }

      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }

  pollRun("{{ run_id }}");
</script>
{% endif %}
</body>
</html>
"""


def default_form_values():
    return {
        "model_key": "unconditional",
        "n_images": 4,
        "batch_size": 4,
        "inference_steps": 50,
    }


@app.route("/", methods=["GET"])
def index():
    return render_template_string(
        HTML_TEMPLATE,
        model_choices=MODEL_CHOICES,
        result=None,
        error=None,
        form_values=default_form_values(),
        param_limits=PARAM_LIMITS,
        run_id=None,
        run_status=None,
        logs="",
    )


@app.route("/generate", methods=["POST"])
def generate():
    form_values = {
        "model_key": request.form.get("model_key", "unconditional"),
        "n_images": request.form.get("n_images", "4"),
        "batch_size": request.form.get("batch_size", "4"),
        "inference_steps": request.form.get("inference_steps", "50"),
    }

    try:
        run_id = generator_app.create_run(
            model_key=form_values["model_key"],
            n_images=int(form_values["n_images"]),
            batch_size=int(form_values["batch_size"]),
            inference_steps=int(form_values["inference_steps"]),
        )

        with RUNS_LOCK:
            run = RUNS[run_id]

        return render_template_string(
            HTML_TEMPLATE,
            model_choices=MODEL_CHOICES,
            result=None,
            error=None,
            form_values=form_values,
            param_limits=PARAM_LIMITS,
            run_id=run_id,
            run_status=run["status"],
            logs=run["logs"],
        )
    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE,
            model_choices=MODEL_CHOICES,
            result=None,
            error=str(e),
            form_values=form_values,
            param_limits=PARAM_LIMITS,
            run_id=None,
            run_status=None,
            logs="",
        ), 500


@app.route("/run_status/<run_id>", methods=["GET"])
def run_status(run_id):
    with RUNS_LOCK:
        run = RUNS.get(run_id)
        if run is None:
            abort(404)
        return {
            "status": run["status"],
            "logs": run["logs"],
            "error": run["error"],
        }


@app.route("/run/<run_id>", methods=["GET"])
def run_result(run_id):
    with RUNS_LOCK:
        run = RUNS.get(run_id)
        if run is None:
            abort(404)

        result = run["result"]
        error = run["error"]
        status = run["status"]
        form_values = run["form_values"]
        logs = run["logs"]

    if result is not None:
        result = dict(result)
        result["preview_url"] = url_for(
            "preview_file",
            run_id=result["run_id"],
            subdir=result["preview_subdir"],
            filename=Path(result["preview_path"]).name,
        )
        result["download_url"] = url_for("download_zip", run_id=result["run_id"])
        result["preview_name"] = Path(result["preview_path"]).name
        result["zip_name"] = Path(result["zip_path"]).name

    return render_template_string(
        HTML_TEMPLATE,
        model_choices=MODEL_CHOICES,
        result=result,
        error=error,
        form_values=form_values,
        param_limits=PARAM_LIMITS,
        run_id=run_id,
        run_status=status,
        logs=logs,
    )


@app.route("/preview/<run_id>/<subdir>/<filename>", methods=["GET"])
def preview_file(run_id, subdir, filename):
    if subdir not in {"images", "overlay"}:
        abort(404)

    path = os.path.join(WEB_OUTPUTS_DIR, run_id, subdir, filename)
    if not os.path.isfile(path):
        abort(404)
    return send_file(path)


@app.route("/download/<run_id>", methods=["GET"])
def download_zip(run_id):
    path = os.path.join(WEB_OUTPUTS_DIR, f"{run_id}_synthetic_data.zip")
    if not os.path.isfile(path):
        abort(404)
    return send_file(path, as_attachment=True, download_name=f"{run_id}_synthetic_data.zip")


def main():
    os.makedirs(WEB_OUTPUTS_DIR, exist_ok=True)
    app.run(host=HOST, port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
