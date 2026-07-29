"""
video_processor.py — Video frame extraction & tensor conversion
================================================================

Extracts uniformly-sampled, centre-cropped, resized frames from a video
file and converts them into the ``(3, T, H, W)`` tensor format expected by
the DMESR ``VideoEncoder``.

Public API:
    extract_frames(video_path, …)         → (num_frames, H, W, 3) uint8 ndarray
    extract_aligned_frames(video_path, …) → same, but between start/end timestamps
    frames_to_tensor(frames)              → (3, num_frames, H, W)  float32 tensor
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import torch

from config import VIDEO_NUM_FRAMES, VIDEO_FRAME_SIZE

logger = logging.getLogger(__name__)

# ImageNet normalisation constants (used by the pretrained ResNet50 inside
# the VideoEncoder pipeline).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _centre_crop_square(frame: np.ndarray) -> np.ndarray:
    """
    Centre-crop *frame* to a square whose side equals ``min(h, w)``.

    Args:
        frame: (H, W, 3) uint8 array.

    Returns:
        Square-cropped (S, S, 3) uint8 array.
    """
    h, w = frame.shape[:2]
    side = min(h, w)
    y_start = (h - side) // 2
    x_start = (w - side) // 2
    return frame[y_start : y_start + side, x_start : x_start + side]


def _uniform_sample_indices(total_frames: int, num_samples: int) -> list[int]:
    """
    Return *num_samples* evenly-spaced indices in ``[0, total_frames)``.

    If the video has fewer frames than requested, indices are repeated so
    that the output always has exactly *num_samples* entries.
    """
    if total_frames <= 0:
        return [0] * num_samples
    if total_frames <= num_samples:
        # Repeat last frame to pad
        indices = list(range(total_frames))
        while len(indices) < num_samples:
            indices.append(total_frames - 1)
        return indices
    step = total_frames / num_samples
    return [int(i * step) for i in range(num_samples)]


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    num_frames: int = VIDEO_NUM_FRAMES,
    frame_size: int = VIDEO_FRAME_SIZE,
) -> np.ndarray:
    """
    Extract *num_frames* uniformly-sampled frames from a video file.

    Each frame is centre-cropped to a square and resized to
    ``(frame_size, frame_size)``.

    Args:
        video_path:  Absolute path to the video file.
        num_frames:  How many frames to sample (default 8).
        frame_size:  Output spatial resolution (default 112).

    Returns:
        ``(num_frames, frame_size, frame_size, 3)`` uint8 ndarray (RGB).

    Raises:
        RuntimeError: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _uniform_sample_indices(total, num_frames)

    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # Use a black frame as fallback
            frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
        else:
            # OpenCV loads BGR — convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = _centre_crop_square(frame)
            frame = cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)

    cap.release()
    return np.stack(frames, axis=0)  # (num_frames, H, W, 3)


def extract_aligned_frames(
    video_path: str,
    start_time: float,
    end_time: float,
    num_frames: int = VIDEO_NUM_FRAMES,
    frame_size: int = VIDEO_FRAME_SIZE,
) -> np.ndarray:
    """
    Extract *num_frames* uniformly between *start_time* and *end_time*.

    Useful when Whisper provides segment-level timestamps and we want
    to align video frames with the corresponding speech segment.

    Args:
        video_path:  Absolute path to the video file.
        start_time:  Segment start in seconds.
        end_time:    Segment end in seconds.
        num_frames:  How many frames to sample (default 8).
        frame_size:  Output spatial resolution (default 112).

    Returns:
        ``(num_frames, frame_size, frame_size, 3)`` uint8 ndarray (RGB).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    segment_len = max(end_frame - start_frame, 1)

    indices = _uniform_sample_indices(segment_len, num_frames)
    # Shift indices into absolute frame positions
    abs_indices = [start_frame + i for i in indices]

    frames: list[np.ndarray] = []
    for idx in abs_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = _centre_crop_square(frame)
            frame = cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)

    cap.release()
    return np.stack(frames, axis=0)  # (num_frames, H, W, 3)


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """
    Convert a ``(T, H, W, 3)`` uint8 RGB array to a normalised float
    tensor in the layout ``(3, T, H, W)`` expected by the DMESR
    ``VideoEncoder``.

    Normalisation uses ImageNet mean/std because the downstream ResNet50
    backbone was pretrained on ImageNet.

    Args:
        frames: ``(T, H, W, 3)`` uint8 ndarray.

    Returns:
        ``(3, T, H, W)`` float32 ``torch.Tensor``.
    """
    # float32, [0, 1]
    f = frames.astype(np.float32) / 255.0  # (T, H, W, 3)

    # ImageNet normalisation per-pixel
    f = (f - _IMAGENET_MEAN) / _IMAGENET_STD  # still (T, H, W, 3)

    # (T, H, W, 3) → (T, 3, H, W) → (3, T, H, W)
    t = torch.from_numpy(f).permute(0, 3, 1, 2)  # (T, 3, H, W)
    t = t.permute(1, 0, 2, 3)                     # (3, T, H, W)
    return t
