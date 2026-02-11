"""
Native macOS Training Script with MPS Acceleration
Run this with: python train_native.py
"""

import sys
from pathlib import Path
import torch
import torchaudio
from speechbrain.inference import SpeakerRecognition
import time

# Add Enkidu to path
ENKIDU_PATH = Path(__file__).parent / 'dependencies' / 'Enkidu'
sys.path.insert(0, str(ENKIDU_PATH))

from core import Enkidu

print("="*60)
print("ENKIDU TRAINING - Native macOS with MPS")
print("="*60)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Device selection - prioritize MPS (Apple Silicon GPU)
if torch.backends.mps.is_available():
    device = 'mps'
    print("✅ Using MPS (Apple M4 GPU)")
elif torch.cuda.is_available():
    device = 'cuda'
    print("✅ Using CUDA GPU")
else:
    device = 'cpu'
    print("⚠️  Using CPU (slower)")

print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}\n")

CONFIG = {
    'num_training_samples': 20,
    'sample_rate': 16000,
    'steps': 30,
    'alpha': 0.1,
    'mask_ratio': 0.3,
    'frame_length': 30,
    'noise_level': 0.4,
    'device': device
}

# ============================================================================
# CREATE SYNTHETIC TRAINING DATA
# ============================================================================

print("="*60)
print("CREATING TRAINING DATA")
print("="*60)
print("Using synthetic audio samples for demonstration...")

# Create synthetic samples (simulating voice recordings)
training_waveforms = []
for i in range(CONFIG['num_training_samples']):
    # Generate 3-second audio samples
    duration = 3  # seconds
    num_samples = CONFIG['sample_rate'] * duration
    
    # Create synthetic waveform (simulates speech)
    waveform = torch.randn(1, num_samples) * 0.1
    training_waveforms.append(waveform)

print(f"✓ Created {len(training_waveforms)} synthetic audio samples")
print(f"  Duration: {duration}s each")
print(f"  Sample rate: {CONFIG['sample_rate']} Hz\n")

# ============================================================================
# INITIALIZE MODELS
# ============================================================================

print("="*60)
print("INITIALIZING MODELS")
print("="*60)
print("Loading SpeechBrain speaker recognition model...")

speaker_model = SpeakerRecognition.from_hparams(
    'speechbrain/spkrec-ecapa-voxceleb',
    run_opts={"device": device}
)
print("✓ Speaker recognition model loaded")

print("Initializing Enkidu model...")
enkidu = Enkidu(
    model=speaker_model,
    steps=CONFIG['steps'],
    alpha=CONFIG['alpha'],
    mask_ratio=CONFIG['mask_ratio'],
    frame_length=CONFIG['frame_length'],
    noise_level=CONFIG['noise_level'],
    device=device
)
print("✓ Enkidu model initialized\n")

# ============================================================================
# TRAINING
# ============================================================================

print("="*60)
print("TRAINING NOISE PATTERNS")
print("="*60)
print(f"Configuration:")
print(f"  Samples: {len(training_waveforms)}")
print(f"  Steps: {CONFIG['steps']}")
print(f"  Noise level: {CONFIG['noise_level']}")
print(f"  Device: {device}")
print(f"\nTraining started... (estimated 3-5 minutes with MPS)\n")

start_time = time.time()

# Train universal noise patterns
noise_real, noise_imag = enkidu(training_waveforms)

elapsed = time.time() - start_time

print(f"\n{'='*60}")
print("✅ TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"Training time: {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)")
print(f"Noise shape: {noise_real.shape}\n")

# ============================================================================
# TEST PROTECTION
# ============================================================================

print("="*60)
print("TESTING PROTECTION")
print("="*60)

# Use first training sample for testing
test_audio = training_waveforms[0].to(device)
print(f"Test audio: {test_audio.shape[1] / CONFIG['sample_rate']:.1f} seconds")

# Apply protection
protected = enkidu.add_noise(
    test_audio,
    noise_real,
    noise_imag,
    mask_ratio=0.3,
    random_offset=False,
    noise_smooth=True
)
print("✓ Protection applied")

# Save test audio files
output_dir = Path(__file__).parent
torchaudio.save(str(output_dir / 'test_original.flac'), test_audio.cpu(), CONFIG['sample_rate'])
torchaudio.save(str(output_dir / 'test_protected.flac'), protected.cpu(), CONFIG['sample_rate'])
print("✓ Test files saved:")
print(f"  - test_original.flac")
print(f"  - test_protected.flac")

# Evaluate protection effectiveness
print("\nEvaluating protection...")
with torch.no_grad():
    orig_emb = speaker_model.encode_batch(test_audio)
    prot_emb = speaker_model.encode_batch(protected)
    similarity = torch.nn.functional.cosine_similarity(
        orig_emb, prot_emb, dim=-1
    ).item()

print(f"\n📊 EVALUATION RESULTS")
print(f"{'='*60}")
print(f"Similarity Score: {similarity:.4f} (lower = better)")

if similarity < 0.5:
    print("Protection Level: ✅ EXCELLENT")
elif similarity < 0.7:
    print("Protection Level: ✅ GOOD")
elif similarity < 0.85:
    print("Protection Level: ⚠️  MODERATE")
else:
    print("Protection Level: ⚠️  WEAK")

print()

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("="*60)
print("SAVING RESULTS")
print("="*60)

# Save noise patterns
checkpoint = {
    'noise_real': noise_real.cpu(),
    'noise_imag': noise_imag.cpu(),
    'config': CONFIG,
    'metadata': {
        'device': device,
        'training_time_seconds': elapsed,
        'similarity_score': similarity,
        'pytorch_version': torch.__version__,
        'sample_type': 'synthetic'
    }
}

output_path = output_dir / 'noise_patterns.pt'
torch.save(checkpoint, output_path)
print(f"✓ Noise patterns saved: {output_path}")

# Summary
print(f"\n{'='*60}")
print("✅ ALL DONE!")
print(f"{'='*60}")
print(f"\nSummary:")
print(f"  Device used: {device}")
print(f"  Training time: {elapsed/60:.2f} minutes")
print(f"  Protection score: {similarity:.4f}")
print(f"  Output: noise_patterns.pt")
print(f"\nNext steps:")
print(f"  1. Use noise_patterns.pt with your pipeline")
print(f"  2. Test with: python src/enkidu_experiments/enkidu_pipeline.py")
print(f"\nExample usage:")
print(f"  from src.enkidu_experiments.enkidu_pipeline import EnkiduPipeline")
print(f"  pipeline = EnkiduPipeline()")
print(f"  noise_real, noise_imag = pipeline.load_noise('noise_patterns.pt')")
print(f"  pipeline.protect_audio_file('input.flac', 'output.flac', noise_real, noise_imag)")