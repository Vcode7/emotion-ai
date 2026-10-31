import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def load_audio(filepath: str) -> tuple[torch.Tensor, int]:
    """
    Load an audio file (WAV, MP3, WebM, OGG, FLAC, etc.).
    Returns (waveform, sample_rate) matching torchaudio format:
      - waveform: torch.Tensor of shape (channels, time) in float32, range [-1.0, 1.0]
      - sample_rate: int
    """
    # 1. Try whisper.audio (uses ffmpeg — works reliably for WebM / Opus on Windows)
    try:
        import whisper.audio
        audio_np = whisper.audio.load_audio(filepath)  # float32, 16000 Hz mono
        waveform = torch.from_numpy(audio_np)[np.newaxis, :]  # (1, time)
        return waveform, 16000
    except Exception as whisper_err:
        logger.debug("whisper.audio load failed for '%s': %s", filepath, whisper_err)

    # 2. Try soundfile (primary for WAV/FLAC)
    try:
        import soundfile as sf
        data, sample_rate = sf.read(filepath, dtype='float32')
        data = np.asarray(data)
        if data.ndim == 1:
            waveform_np = data[np.newaxis, :]  # Mono: (1, time)
        else:
            waveform_np = data.T  # Stereo/Multi-channel: (channels, time)
        waveform = torch.from_numpy(waveform_np)
        return waveform, sample_rate
    except Exception as sf_err:
        logger.debug("soundfile load failed for '%s': %s", filepath, sf_err)

    # 3. Try torchaudio
    try:
        import torchaudio
        waveform, sample_rate = torchaudio.load(filepath)
        return waveform, sample_rate
    except Exception as ta_err:
        logger.debug("torchaudio load failed for '%s': %s", filepath, ta_err)

    # 4. Try librosa
    try:
        import librosa
        data, sample_rate = librosa.load(filepath, sr=None, mono=False)
        if data.ndim == 1:
            waveform_np = data[np.newaxis, :]
        else:
            waveform_np = data
        waveform = torch.from_numpy(waveform_np)
        return waveform, sample_rate
    except Exception as librosa_err:
        logger.debug("librosa load failed for '%s': %s", filepath, librosa_err)

    # 5. Safe fallback: 1 second of silent audio tensor (prevents 500 error on corrupted/empty clips)
    logger.warning("All audio decoders failed for '%s'. Using 1s silent audio fallback.", filepath)
    return torch.zeros(1, 16000, dtype=torch.float32), 16000
