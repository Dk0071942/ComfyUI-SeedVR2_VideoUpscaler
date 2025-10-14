"""SeedVR2 Gradio interface for image, video, and batch upscaling."""

import os
import subprocess

# Ensure torch uses async allocator like CLI (must run before importing torch)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

import sys
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

# Add project root to sys.path for relative imports when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.downloads import download_weight, get_base_cache_dir  # noqa: E402
from src.core.model_manager import configure_runner  # noqa: E402
from src.core.generation import generation_loop  # noqa: E402


# ---------------------------------------------------------------------------
# Runner management
# ---------------------------------------------------------------------------
class RunnerManager:
    """Simple runner cache to avoid reloading weights for every request."""

    _cache = {}
    _lock = threading.Lock()

    @classmethod
    def _make_key(cls, model_name: str, preserve_vram: bool, debug: bool) -> Tuple[str, bool, bool]:
        return (model_name, bool(preserve_vram), bool(debug))

    @classmethod
    def get_runner(cls, model_name: str, preserve_vram: bool = False, debug: bool = False):
        key = cls._make_key(model_name, preserve_vram, debug)
        with cls._lock:
            cached = cls._cache.get(key)
            if cached is not None:
                return cached

            base_cache_dir = get_base_cache_dir()
            runner = configure_runner(
                model=model_name,
                base_cache_dir=base_cache_dir,
                preserve_vram=preserve_vram,
                debug=debug,
                block_swap_config=None,
                cached_runner=None,
            )
            cls._cache[key] = runner
            return runner

    @classmethod
    def clear_cache(cls) -> str:
        with cls._lock:
            for runner in cls._cache.values():
                try:
                    if hasattr(runner, "dit") and runner.dit is not None:
                        runner.dit.to("cpu")
                    if hasattr(runner, "vae") and runner.vae is not None:
                        runner.vae.to("cpu")
                except Exception:
                    pass
            cls._cache.clear()
        torch.cuda.empty_cache()
        return "Cleared cached runners and released GPU memory."


INFERENCE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _env_flag(name: str, default: bool = False) -> bool:
    """Return True if the environment variable is set to a truthy value."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_basic_auth() -> Optional[List[Tuple[str, str]]]:
    """Build a credentials list for Gradio basic auth based on environment variables."""
    if not _env_flag("GRADIO_AUTH_ENABLED", default=False):
        return None

    username = os.getenv("GRADIO_AUTH_USERNAME", "").strip()
    password = os.getenv("GRADIO_AUTH_PASSWORD", "")

    if not username or not password:
        print("⚠️ Gradio auth enabled but username or password missing; skipping authentication.")
        return None

    return [(username, password)]


def _prepare_image_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array)
    tensor = tensor.unsqueeze(0).to(torch.float16)  # [1, H, W, C]
    return tensor


def _tensor_to_pil(frame: torch.Tensor) -> Image.Image:
    array = frame.clamp(0.0, 1.0).cpu().numpy()
    array = (array * 255.0).astype(np.uint8)
    return Image.fromarray(array)


def _extract_frames(
    video_path: str,
    skip_first_frames: int = 0,
    load_cap: int = 0,
    progress: Optional[gr.Progress] = None,
) -> Tuple[torch.Tensor, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: List[np.ndarray] = []

    idx = 0
    loaded = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx < skip_first_frames:
            idx += 1
            continue

        if load_cap > 0 and loaded >= load_cap:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb.astype(np.float32) / 255.0)
        idx += 1
        loaded += 1

        if progress and total_frames > 0:
            progress(min(loaded / max(total_frames, 1), 0.9), desc=f"Reading frames {loaded}/{total_frames}")

    cap.release()

    if not frames:
        raise ValueError("No frames extracted from video.")

    stacked = torch.from_numpy(np.stack(frames)).to(torch.float16)
    return stacked, fps


def _save_video(
    frames: torch.Tensor,
    fps: float,
    audio_source: Optional[str] = None,
    stem: Optional[str] = None,
) -> str:
    if stem:
        tmp_dir = Path(tempfile.mkdtemp(prefix="seedvr2_video_"))
        output_path = tmp_dir / f"{stem}.mp4"
    else:
        tmp_file = tempfile.NamedTemporaryFile(prefix="seedvr2_video_", suffix=".mp4", delete=False)
        tmp_file.close()
        output_path = Path(tmp_file.name)

    frames_np = frames.cpu().numpy()
    frames_np = (frames_np * 255.0).astype(np.uint8)
    T, H, W, _ = frames_np.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError("Failed to open video writer.")

    for idx in range(T):
        bgr = cv2.cvtColor(frames_np[idx], cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()

    if audio_source:
        audio_path = Path(audio_source)
        if audio_path.exists():
            ffmpeg_bin = shutil.which("ffmpeg")
            if ffmpeg_bin:
                tmp_audio = tempfile.NamedTemporaryFile(prefix="seedvr2_audio_", suffix=".mp4", delete=False)
                tmp_audio.close()
                merged_path = Path(tmp_audio.name)
                cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(output_path),
                    "-i",
                    str(audio_path),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a?",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "copy",
                    "-shortest",
                    str(merged_path),
                ]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if result.returncode == 0 and merged_path.exists():
                    try:
                        output_path.unlink(missing_ok=True)
                    except TypeError:
                        if output_path.exists():
                            output_path.unlink()
                    shutil.move(str(merged_path), str(output_path))
                else:
                    print("⚠️ FFmpeg failed to merge audio; returning silent video.")
                    if merged_path.exists():
                        try:
                            merged_path.unlink(missing_ok=True)
                        except TypeError:
                            if merged_path.exists():
                                merged_path.unlink()
            else:
                print("ℹ️ FFmpeg not found; returning silent video.")
        else:
            print("ℹ️ Audio source not found; returning silent video.")

    return str(output_path)


def _save_png_sequence(frames: torch.Tensor, stem: str) -> str:
    tmp_dir = Path(tempfile.mkdtemp(prefix="seedvr2_frames_"))
    output_dir = tmp_dir / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_np = (frames.cpu().numpy() * 255.0).astype(np.uint8)
    total = frames_np.shape[0]
    digits = max(5, len(str(total)))

    for idx, frame in enumerate(frames_np):
        filename = output_dir / f"{stem}_{idx:0{digits}d}.png"
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filename), bgr)

    archive_path = shutil.make_archive(str(output_dir), "zip", output_dir)
    # Clean up extracted frames to keep /tmp tidy
    shutil.rmtree(output_dir, ignore_errors=True)
    return archive_path


def _resolve_input_path(file_input: Any) -> Optional[str]:
    """Return a filesystem path from Gradio file/video inputs."""
    if file_input is None:
        return None
    if isinstance(file_input, str) and file_input:
        return file_input
    if isinstance(file_input, dict):
        return file_input.get("name") or file_input.get("path")
    path_attr = getattr(file_input, "path", None)
    if path_attr:
        return path_attr
    name_attr = getattr(file_input, "name", None)
    if name_attr:
        return name_attr
    return None


def _load_image_input(image_input: Any) -> Tuple[Image.Image, Optional[Path]]:
    """Load an image from Gradio input and return it along with its source path if available."""
    if isinstance(image_input, Image.Image):
        filename = getattr(image_input, "filename", "") or image_input.info.get("filename", "")
        try:
            source_path = Path(filename).expanduser() if filename else None
        except Exception:
            source_path = None
        return image_input, source_path

    resolved_path = _resolve_input_path(image_input)
    if not resolved_path:
        raise gr.Error("Please upload an image to upscale.")

    path = Path(resolved_path).expanduser()
    if not path.exists():
        raise gr.Error(f"Uploaded image not found: {resolved_path}")

    with Image.open(path) as src:
        image = src.copy()

    return image, path


def _run_generation(
    images: torch.Tensor,
    model: str,
    resolution: int,
    cfg_scale: float,
    seed: int,
    batch_size: int,
    preserve_vram: bool,
    temporal_overlap: int = 0,
    debug: bool = False,
    progress: Optional[gr.Progress] = None,
):
    download_weight(model)
    runner = RunnerManager.get_runner(model, preserve_vram=preserve_vram, debug=debug)

    # Ensure cached runner modules return to GPU when VRAM preservation is off
    if not preserve_vram and torch.cuda.is_available():
        blockswap_active = getattr(runner, '_blockswap_active', False)
        if not blockswap_active and runner is not None:
            target_device = torch.device('cuda')
            try:
                current_device = next(runner.dit.parameters()).device
            except StopIteration:
                current_device = target_device
            if current_device != target_device:
                runner.dit = runner.dit.to(target_device)
            if hasattr(runner, 'vae') and runner.vae is not None:
                try:
                    vae_device = next(runner.vae.parameters()).device
                except StopIteration:
                    vae_device = target_device
                if vae_device != target_device:
                    runner.vae = runner.vae.to(target_device)

    def _progress_callback(batch_idx: int, total_batches: int, frames_in_batch: int, message: str = ""):
        if progress and total_batches > 0:
            value = min(batch_idx / total_batches, 1.0)
            desc = message or f"Processing batch {batch_idx}/{total_batches}"
            progress(value, desc=desc)

    with INFERENCE_LOCK:
        result = generation_loop(
            runner=runner,
            images=images,
            cfg_scale=cfg_scale,
            seed=seed,
            res_w=resolution,
            batch_size=batch_size,
            preserve_vram=preserve_vram,
            temporal_overlap=temporal_overlap,
            debug=debug,
            block_swap_config=None,
            progress_callback=_progress_callback,
        )
    return result


# ---------------------------------------------------------------------------
# Image workflow
# ---------------------------------------------------------------------------
def upscale_image(
    image_input: Any,
    model: str,
    resolution: int,
    cfg_scale: float,
    seed: int,
    preserve_vram: bool,
    debug: bool,
):
    if image_input is None:
        raise gr.Error("Please upload an image to upscale.")

    progress = gr.Progress(track_tqdm=True)
    progress(0, desc="Preparing image")

    image, source_path = _load_image_input(image_input)
    tensor = _prepare_image_tensor(image)
    result = _run_generation(
        images=tensor,
        model=model,
        resolution=resolution,
        cfg_scale=cfg_scale,
        seed=seed,
        batch_size=1,
        preserve_vram=preserve_vram,
        temporal_overlap=0,
        debug=debug,
        progress=progress,
    )

    if result.shape[0] == 0:
        raise RuntimeError("Generation returned no frames.")

    output = _tensor_to_pil(result[0])
    stem = source_path.stem if source_path else "upscaled_image"
    tmp_dir = Path(tempfile.mkdtemp(prefix="seedvr2_image_"))
    output_path = tmp_dir / f"{stem}.png"
    output.save(output_path, format="PNG")
    progress(1.0, desc="Completed")
    return output, str(output_path), "Image upscaled successfully."


# ---------------------------------------------------------------------------
# Video workflow
# ---------------------------------------------------------------------------
def upscale_video(
    video_input: Any,
    model: str,
    resolution: int,
    cfg_scale: float,
    seed: int,
    batch_size: int,
    temporal_overlap: int,
    preserve_vram: bool,
    output_format: str,
    skip_first_frames: int,
    frame_cap: int,
    debug: bool,
):
    video_path = _resolve_input_path(video_input)
    if not video_path:
        raise gr.Error("Please provide a video file.")

    progress = gr.Progress(track_tqdm=True)
    progress(0, desc="Extracting frames")

    frames, fps = _extract_frames(video_path, skip_first_frames, frame_cap, progress=progress)
    progress(0.2, desc="Running upscaler")

    result = _run_generation(
        images=frames,
        model=model,
        resolution=resolution,
        cfg_scale=cfg_scale,
        seed=seed,
        batch_size=batch_size,
        preserve_vram=preserve_vram,
        temporal_overlap=temporal_overlap,
        debug=debug,
        progress=progress,
    )

    if result.shape[0] == 0:
        raise RuntimeError("Generation returned no frames.")

    stem = Path(video_path).stem + "_enhanced"

    if output_format == "video":
        progress(0.9, desc="Encoding video")
        video_file = _save_video(result, fps, audio_source=video_path, stem=stem)
        progress(1.0, desc="Completed")
        return video_file, gr.update(value=video_file, visible=True), "Video upscaled successfully."

    progress(0.9, desc="Writing PNG frames")
    archive_path = _save_png_sequence(result, stem)
    progress(1.0, desc="Completed")
    return archive_path, gr.update(value=None, visible=False), "PNG sequence archived successfully."


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
MODEL_OPTIONS = [
    "seedvr2_ema_3b_fp16.safetensors",
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "seedvr2_ema_7b_fp16.safetensors",
    "seedvr2_ema_7b_fp8_e4m3fn.safetensors",
    "seedvr2_ema_7b_sharp_fp16.safetensors",
    "seedvr2_ema_7b_sharp_fp8_e4m3fn.safetensors",
]


def clear_runner_cache():
    return RunnerManager.clear_cache()


def build_interface():
    with gr.Blocks(title="SeedVR2 Upscaler") as demo:
        gr.Markdown(
            """
            # SeedVR2 Upscaler
            Upscale single images or videos using the SeedVR2 diffusion pipeline.
            """
        )

        with gr.Tab("Image Upscaling"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label="Input Image", type="filepath")
                    model_dropdown = gr.Dropdown(MODEL_OPTIONS, value=MODEL_OPTIONS[1], label="Model")
                    resolution_slider = gr.Slider(512, 2048, value=1072, step=16, label="Target short side (px)")
                    cfg_scale_slider = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="CFG Scale")
                    seed_slider = gr.Slider(0, 2**32 - 1, value=100, step=1, label="Seed")
                    preserve_checkbox = gr.Checkbox(value=False, label="Preserve VRAM")
                    debug_checkbox = gr.Checkbox(value=False, label="Enable debug logging")
                    image_button = gr.Button("Upscale Image", variant="primary")
                with gr.Column():
                    image_output = gr.Image(label="Upscaled Image")
                    image_download = gr.File(label="Download PNG")
                    image_status = gr.Textbox(label="Status", interactive=False)

            image_button.click(
                upscale_image,
                inputs=[
                    image_input,
                    model_dropdown,
                    resolution_slider,
                    cfg_scale_slider,
                    seed_slider,
                    preserve_checkbox,
                    debug_checkbox,
                ],
                outputs=[image_output, image_download, image_status],
            )

        with gr.Tab("Video Upscaling"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Input Video", sources=["upload"])
                    model_dropdown_v = gr.Dropdown(MODEL_OPTIONS, value=MODEL_OPTIONS[1], label="Model")
                    resolution_slider_v = gr.Slider(512, 2048, value=1072, step=16, label="Target short side (px)")
                    cfg_scale_slider_v = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="CFG Scale")
                    seed_slider_v = gr.Slider(0, 2**32 - 1, value=100, step=1, label="Seed")
                    batch_size_slider_v = gr.Slider(1, 65, value=5, step=4, label="Batch size (use 4n+1 for best results)")
                    temporal_overlap_slider = gr.Slider(0, 12, value=0, step=1, label="Temporal overlap")
                    output_format_radio = gr.Radio(["video", "png"], value="video", label="Output format")
                    skip_frames_slider = gr.Slider(0, 100, value=0, step=1, label="Skip first N frames")
                    frame_cap_slider = gr.Slider(0, 2000, value=0, step=10, label="Max frames (0 = all)")
                    preserve_checkbox_v = gr.Checkbox(value=False, label="Preserve VRAM")
                    debug_checkbox_v = gr.Checkbox(value=False, label="Enable debug logging")
                    video_button = gr.Button("Upscale Video", variant="primary")
                with gr.Column():
                    video_file_output = gr.File(label="Download result")
                    video_preview_output = gr.Video(label="Preview", visible=False)
                    video_status = gr.Textbox(label="Status", interactive=False)

            video_button.click(
                upscale_video,
                inputs=[
                    video_input,
                    model_dropdown_v,
                    resolution_slider_v,
                    cfg_scale_slider_v,
                    seed_slider_v,
                    batch_size_slider_v,
                    temporal_overlap_slider,
                    preserve_checkbox_v,
                    output_format_radio,
                    skip_frames_slider,
                    frame_cap_slider,
                    debug_checkbox_v,
                ],
                outputs=[video_file_output, video_preview_output, video_status],
            )

        with gr.Accordion("Utilities", open=False):
            clear_button = gr.Button("Clear cached models")
            clear_status = gr.Textbox(label="Cache status", interactive=False)
            clear_button.click(clear_runner_cache, outputs=clear_status)

    return demo


def main():
    demo = build_interface()
    launch_kwargs: dict[str, Any] = {}

    server_name = os.getenv("GRADIO_SERVER_NAME", "").strip() or "127.0.0.1"
    launch_kwargs["server_name"] = server_name

    port_raw = os.getenv("GRADIO_SERVER_PORT", "7860").strip()
    try:
        launch_kwargs["server_port"] = int(port_raw)
    except ValueError:
        print(f"⚠️ Invalid GRADIO_SERVER_PORT '{port_raw}', defaulting to 7860.")
        launch_kwargs["server_port"] = 7860

    auth_credentials = _resolve_basic_auth()
    if auth_credentials:
        launch_kwargs["auth"] = auth_credentials

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
