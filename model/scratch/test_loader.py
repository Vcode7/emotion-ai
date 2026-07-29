import os
import sys
import wave
import struct
import math
from pathlib import Path
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.audio_loader import load_audio

def generate_dummy_wav(path, duration=1.0, sample_rate=16000):
    num_samples = int(duration * sample_rate)
    frequency = 440.0
    with wave.open(path, 'w') as w:
        w.setnchannels(1) # mono
        w.setsampwidth(2) # 16-bit
        w.setframerate(sample_rate)
        for i in range(num_samples):
            value = int(32767.0 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
            data = struct.pack('<h', value)
            w.writeframesraw(data)

def main():
    test_file = "dummy_test.wav"
    try:
        print("Generating dummy WAV file...")
        generate_dummy_wav(test_file)
        
        print("Loading audio using custom loader...")
        waveform, sr = load_audio(test_file)
        
        print("\nLoaded Details:")
        print(f"  Waveform type:  {type(waveform)}")
        print(f"  Waveform shape: {waveform.shape}")
        print(f"  Waveform dtype: {waveform.dtype}")
        print(f"  Sample rate:    {sr}")
        
        # Assertions to verify exact API compatibility with torchaudio.load
        assert isinstance(waveform, torch.Tensor), "Output must be a torch.Tensor"
        assert waveform.shape[0] == 1, "Should be mono (1 channel)"
        assert waveform.ndim == 2, "Should be 2D shape (channels, time)"
        assert sr == 16000, "Sample rate should be 16000"
        
        print("\n[OK] Loader test passed successfully!")
    except Exception as e:
        print(f"\n[FAIL] Loader test failed: {e}")
        sys.exit(1)
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    main()
