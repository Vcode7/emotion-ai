"""
inference.py — DMESR Inference Pipeline
========================================

Manages per-session DMESR inference instances so that each conversation
maintains its own EmotionMemory and temporal context.

Architecture:
    InferenceManager (singleton)
        └─ DMESR model (loaded once, shared)
        └─ sessions: dict[session_id → DMESRInference]

Each public method accepts a session_id and the relevant input tensors,
delegates to DMESRInference.process_sentence(), and returns the full
emotion output dict.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Ensure the project root is importable so `from model.dmesr import ...`
#  works regardless of CWD.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.dmesr import (  # noqa: E402
    DMESR,
    DMESRConfig,
    DMESRInference,
)
from config import DEVICE, AUDIO_SAMPLE_RATE, VIDEO_NUM_FRAMES, VIDEO_FRAME_SIZE  # noqa: E402


class InferenceManager:
    """
    Singleton-style manager that:
      1. Loads the DMESR model once during startup.
      2. Creates / retrieves per-session DMESRInference wrappers.
      3. Provides convenience methods for each input modality combination.

    All methods return the dict produced by ``DMESRInference.process_sentence()``.
    """

    # ------------------------------------------------------------------
    #  Construction & model loading
    # ------------------------------------------------------------------

    def __init__(self, device: str = DEVICE) -> None:
        """
        Load the DMESR model onto *device* and prepare session storage.

        The model is put into eval mode and gradient computation is
        disabled globally for inference speed.
        """
        self.device = device
        self.cfg = DMESRConfig()

        # Resolve pickling/unpickling issues by mapping DMESRConfig to __main__
        import sys
        sys.modules['__main__'].DMESRConfig = DMESRConfig

        # Instantiate the full DMESR model (all sub-modules)
        self.model = DMESR(self.cfg)

        # Load the checkpoint
        ckpt_path = Path(_PROJECT_ROOT) / "model" / "checkpoint" / "dmesr_mosei_best.pt"
        if ckpt_path.exists():
            try:
                checkpoint = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                logger.info("Loaded DMESR model checkpoint from %s (strict=False)", ckpt_path)
            except Exception as e:
                logger.error("Failed to load DMESR model checkpoint: %s. Running with random initialization.", e)
        else:
            logger.warning("No checkpoint found at %s. Running with random initialization.", ckpt_path)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Per-session inference wrappers, keyed by session_id
        self._sessions: Dict[str, DMESRInference] = {}

        self._model_loaded = True

    # ------------------------------------------------------------------
    #  Session management
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> DMESRInference:
        """
        Return the ``DMESRInference`` wrapper for *session_id*.

        A new wrapper (with a fresh ``EmotionMemory``) is created
        transparently when the session is seen for the first time.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = DMESRInference(
                model=self.model,
                cfg=self.cfg,
                device=self.device,
            )
        return self._sessions[session_id]

    @property
    def session_count(self) -> int:
        """Number of active sessions currently tracked."""
        return len(self._sessions)

    @property
    def model_loaded(self) -> bool:
        """Whether the DMESR model was loaded successfully."""
        return self._model_loaded

    # ------------------------------------------------------------------
    #  Modality-specific helpers
    # ------------------------------------------------------------------

    def _zero_audio(self) -> torch.Tensor:
        """Return a silent audio waveform (4 s at 16 kHz)."""
        return torch.zeros(AUDIO_SAMPLE_RATE * 4, dtype=torch.float32)

    def _zero_video(self) -> torch.Tensor:
        """Return a blank video tensor (3, 8, 112, 112)."""
        return torch.zeros(
            3, VIDEO_NUM_FRAMES, VIDEO_FRAME_SIZE, VIDEO_FRAME_SIZE,
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    #  Public inference methods
    # ------------------------------------------------------------------

    def process_text_only(self, session_id: str, text: str) -> Dict:
        """
        Process a text-only user turn.

        Audio and video channels are filled with zeros so the fusion
        transformer still receives all three modality tokens.
        """
        session = self.get_session(session_id)
        return session.process_sentence(
            text=text,
            audio_waveform=self._zero_audio(),
            video_frames=self._zero_video(),
        )

    def process_with_audio(
        self, session_id: str, text: str, audio_waveform: torch.Tensor
    ) -> Dict:
        """
        Process a turn with real audio and zero-filled video.

        Args:
            audio_waveform: 1-D float tensor of raw 16 kHz samples.
        """
        session = self.get_session(session_id)
        return session.process_sentence(
            text=text,
            audio_waveform=audio_waveform,
            video_frames=self._zero_video(),
        )

    def process_with_video(
        self, session_id: str, text: str, video_frames_tensor: torch.Tensor
    ) -> Dict:
        """
        Process a turn with real video frames and zero-filled audio.

        Args:
            video_frames_tensor: (3, 8, 112, 112) normalised float tensor.
        """
        session = self.get_session(session_id)
        return session.process_sentence(
            text=text,
            audio_waveform=self._zero_audio(),
            video_frames=video_frames_tensor,
        )

    def process_full(
        self,
        session_id: str,
        text: str,
        audio_waveform: torch.Tensor,
        video_frames_tensor: torch.Tensor,
    ) -> Dict:
        """
        Process a turn with all three modalities.

        Args:
            audio_waveform:      1-D float tensor (16 kHz raw samples).
            video_frames_tensor: (3, 8, 112, 112) normalised float tensor.
        """
        session = self.get_session(session_id)
        return session.process_sentence(
            text=text,
            audio_waveform=audio_waveform,
            video_frames=video_frames_tensor,
        )

    # ------------------------------------------------------------------
    #  Session lifecycle
    # ------------------------------------------------------------------

    def reset_session(self, session_id: str) -> None:
        """
        Clear the emotion memory for *session_id*.

        The session wrapper is removed entirely; a fresh one will be
        created on the next call.
        """
        if session_id in self._sessions:
            self._sessions[session_id].reset_conversation()
            del self._sessions[session_id]
