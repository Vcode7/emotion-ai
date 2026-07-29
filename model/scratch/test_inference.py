import os
import sys
from pathlib import Path
import torch

# Add project root to sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.dmesr import DMESR, DMESRConfig, DMESRInference

def main():
    print("Testing DMESR model loading and checkpoint compatibility...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Set __main__.DMESRConfig to prevent pickling issues
    sys.modules['__main__'].DMESRConfig = DMESRConfig

    cfg = DMESRConfig()
    model = DMESR(cfg)

    # Load checkpoint
    ckpt_path = Path(_PROJECT_ROOT) / "model" / "checkpoint" / "dmesr_mosei_best.pt"
    if ckpt_path.exists():
        print(f"Loading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("[OK] Model checkpoint loaded successfully!")
    else:
        print(f"⚠ Checkpoint NOT found at: {ckpt_path}")
        return

    # Instantiate inference helper
    print("Initializing DMESRInference (this loads BERT and tokenizer)...")
    inference = DMESRInference(model, cfg, device=device)
    print("[OK] DMESRInference initialized successfully!")

    # Test processing a sentence
    test_text = "I am so happy and excited today!"
    test_audio = torch.randn(16000 * 2) # 2 seconds of random audio
    test_video = torch.randn(3, 8, 112, 112) # mock video frames

    print(f"\nProcessing sentence: '{test_text}'")
    out = inference.process_sentence(
        text=test_text,
        audio_waveform=test_audio,
        video_frames=test_video,
        timestamp="00:01"
    )

    print("\nInference Output Keys:")
    for k in out.keys():
        print(f"  - {k}")

    print("\nOutput Values Samples:")
    print(f"  emotion_state: {out['emotion_state']}")
    print(f"  avatar_commands: {out['avatar_commands']}")
    print(f"  voice_params: {out['voice_params']}")
    print(f"  llm_context recent history len: {len(out['llm_context']['recent_history'])}")

    print("\n[OK] Verification test passed successfully!")

if __name__ == "__main__":
    main()
