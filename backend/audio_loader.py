import numpy as np
import torch

def load_audio(filepath: str) -> tuple[torch.Tensor, int]:
    """
    Load an audio file using soundfile as the primary backend and librosa as a fallback.
    Returns (waveform, sample_rate) matching the format of torchaudio.load():
      - waveform: torch.Tensor of shape (channels, time) in float32, range [-1.0, 1.0]
      - sample_rate: int
    """
    # 1. Try soundfile first (primary)
    try:
        import soundfile as sf
        # Read as float32 in [-1.0, 1.0]. Shape is (time,) for mono or (time, channels) for stereo
        data, sample_rate = sf.read(filepath, dtype='float32')
        
        # Ensure it is a numpy array
        data = np.asarray(data)
        
        # Reshape to (channels, time) to match torchaudio.load shape format
        if data.ndim == 1:
            waveform_np = data[np.newaxis, :]  # Mono: (1, time)
        else:
            waveform_np = data.T  # Stereo/Multi-channel: (channels, time)
            
        waveform = torch.from_numpy(waveform_np)
        return waveform, sample_rate
        
    except Exception as sf_err:
        # 2. Fallback to librosa if soundfile fails
        try:
            import librosa
            # sr=None preserves original sample rate, mono=False preserves channels
            data, sample_rate = librosa.load(filepath, sr=None, mono=False)
            
            # librosa returns (time,) for mono, (channels, time) for stereo
            if data.ndim == 1:
                waveform_np = data[np.newaxis, :]  # Mono: (1, time)
            else:
                waveform_np = data  # Stereo/Multi-channel: (channels, time)
                
            waveform = torch.from_numpy(waveform_np)
            return waveform, sample_rate
            
        except Exception as librosa_err:
            raise RuntimeError(
                f"Failed to load audio file '{filepath}' with both soundfile and librosa.\n"
                f"soundfile error: {sf_err}\n"
                f"librosa error: {librosa_err}"
            )
