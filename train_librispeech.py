"""
Native macOS Training Script with LibriSpeech Dataset
Uses torchaudio.datasets.LIBRISPEECH for reliable downloads
"""

import sys
from pathlib import Path
import torch
import torchaudio
from speechbrain.inference import SpeakerRecognition
from collections import defaultdict
import time

# Add Enkidu to path
ENKIDU_PATH = Path(__file__).parent / 'dependencies' / 'Enkidu'
sys.path.insert(0, str(ENKIDU_PATH))

from core import Enkidu

print("="*60)
print("ENKIDU TRAINING - LibriSpeech Dataset")
print("="*60)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Device selection - prioritize MPS
if torch.backends.mps.is_available():
    device = 'mps'
    print("✅ Using MPS (Apple M4 GPU)")
else:
    device = 'cpu'
    print("⚠️  Using CPU")

print(f"PyTorch: {torch.__version__}")
print(f"Device: {device}\n")

CONFIG = {
    'num_training_samples': 20,
    'sample_rate': 16000,
    'steps': 30,
    'alpha': 0.1,
    'mask_ratio': 0.3,
    'frame_length': 30,
    'noise_level': 0.4,
    'device': device,
}

# ============================================================================
# LOAD LIBRISPEECH DATASET
# ============================================================================

print("="*60)
print("LOADING LIBRISPEECH DATASET")
print("="*60)

# Set up data directory
DATA_DIR = Path(__file__).parent / 'data' / 'librispeech'
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"Data directory: {DATA_DIR}")
print("Downloading LibriSpeech dev-clean subset...")
print("(First run will download ~350MB, subsequent runs use cached data)\n")

try:
    # Use torchaudio's LIBRISPEECH dataset - it handles downloads properly
    dataset = torchaudio.datasets.LIBRISPEECH(
        root=str(DATA_DIR),
        url="dev-clean",  # Small validation set: ~2.7K samples, ~350MB
        download=True
    )
    
    print(f"✓ Loaded {len(dataset)} samples from LibriSpeech dev-clean")
    print(f"  Dataset cached at: {DATA_DIR}\n")
    
except Exception as e:
    print(f"⚠️  Failed to load LibriSpeech: {e}")
    print("Falling back to synthetic data...\n")
    
    # Fallback to synthetic
    dataset = []
    for i in range(100):
        fake_sample = (
            torch.randn(1, CONFIG['sample_rate'] * 3) * 0.1,  # waveform
            CONFIG['sample_rate'],  # sample_rate
            "",  # transcript
            i % 10,  # speaker_id
            0,  # chapter_id
            0   # utterance_id
        )
        dataset.append(fake_sample)
    print(f"✓ Created {len(dataset)} synthetic samples\n")

# ============================================================================
# ORGANIZE BY SPEAKER
# ============================================================================

print("="*60)
print("ORGANIZING SAMPLES BY SPEAKER")
print("="*60)

# Group samples by speaker
# torchaudio LIBRISPEECH returns tuples: (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)
speaker_samples = defaultdict(list)
for sample in dataset:
    # Extract speaker_id (index 3 in the tuple)
    speaker_id = sample[3] if len(sample) > 3 else 0
    speaker_samples[speaker_id].append(sample)

# Find speaker with most samples
if len(speaker_samples) > 0:
    # Get speakers with enough samples
    valid_speakers = {
        spk: samples for spk, samples in speaker_samples.items()
        if len(samples) >= CONFIG['num_training_samples']
    }
    
    if len(valid_speakers) > 0:
        # Select first valid speaker
        selected_speaker = list(valid_speakers.keys())[0]
        selected_samples = valid_speakers[selected_speaker][:CONFIG['num_training_samples']]
        print(f"✓ Selected speaker: {selected_speaker}")
    else:
        # Not enough samples per speaker, just take first N
        print("⚠️  Not enough samples per speaker")
        selected_samples = list(dataset)[:CONFIG['num_training_samples']]
        print(f"✓ Using first {len(selected_samples)} samples (mixed speakers)")
else:
    # No speaker organization, just take first N
    selected_samples = list(dataset)[:CONFIG['num_training_samples']]
    print(f"✓ Using first {len(selected_samples)} samples")

print(f"✓ Total samples for training: {len(selected_samples)}\n")

# ============================================================================
# PROCESS AUDIO SAMPLES
# ============================================================================

print("="*60)
print("PROCESSING AUDIO SAMPLES")
print("="*60)

training_waveforms = []
for i, sample in enumerate(selected_samples):
    try:
        # torchaudio LIBRISPEECH returns: (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)
        waveform = sample[0]  # Already loaded as tensor!
        sample_rate = sample[1]
        
        # Ensure mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != CONFIG['sample_rate']:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, CONFIG['sample_rate']
            )
        
        training_waveforms.append(waveform)
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{len(selected_samples)} samples")
            
    except Exception as e:
        print(f"  ⚠️  Skipped sample {i}: {e}")
        continue

print(f"\n✓ Successfully processed {len(training_waveforms)} samples")

# Show sample info
if len(training_waveforms) > 0:
    durations = [w.shape[1] / CONFIG['sample_rate'] for w in training_waveforms]
    print(f"  Duration range: {min(durations):.1f}s - {max(durations):.1f}s")
    print(f"  Average: {sum(durations)/len(durations):.1f}s\n")
else:
    print("⚠️  No samples processed! Creating synthetic data...")
    for i in range(CONFIG['num_training_samples']):
        waveform = torch.randn(1, CONFIG['sample_rate'] * 3) * 0.1
        training_waveforms.append(waveform)
    print(f"✓ Created {len(training_waveforms)} synthetic samples\n")

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
torchaudio.save(str(output_dir / 'test_original_librispeech.flac'), test_audio.cpu(), CONFIG['sample_rate'])
torchaudio.save(str(output_dir / 'test_protected_librispeech.flac'), protected.cpu(), CONFIG['sample_rate'])
print("✓ Test files saved:")
print(f"  - test_original_librispeech.flac")
print(f"  - test_protected_librispeech.flac")

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
        'dataset': 'LibriSpeech',
        'dataset_split': 'dev-clean',
        'num_samples': len(training_waveforms),
        'speaker_id': selected_speaker if 'selected_speaker' in locals() else 'mixed'
    }
}

output_path = output_dir / 'noise_patterns_librispeech.pt'
torch.save(checkpoint, output_path)
print(f"✓ Noise patterns saved: {output_path}")

# Summary
print(f"\n{'='*60}")
print("✅ ALL DONE!")
print(f"{'='*60}")
print(f"\nSummary:")
print(f"  Dataset: LibriSpeech (dev-clean)")
print(f"  Device: {device}")
print(f"  Training time: {elapsed/60:.2f} minutes")
print(f"  Protection score: {similarity:.4f}")
print(f"  Output: noise_patterns_librispeech.pt")
print(f"\nNext steps:")
print(f"  1. Use with your pipeline:")
print(f"     pipeline.load_noise('noise_patterns_librispeech.pt')")
print(f"  2. Compare with synthetic version")
print(f"  3. Test on real audio files")