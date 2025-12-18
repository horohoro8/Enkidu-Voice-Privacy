#!/usr/bin/env python3
"""
Test the Enkidu pipeline with podcast audio.

Usage:
    python test_podcast.py <audio_file>
"""

import sys
from pathlib import Path
import torchaudio

# Add src to path so we can import the pipeline
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enkidu_experiments.enkidu_pipeline import EnkiduPipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_podcast.py <audio_file>")
        print("\nExample:")
        print("  python test_podcast.py audio_samples/podcast_cropped_300s.wav")
        sys.exit(1)
    
    input_audio = Path(sys.argv[1])
    
    if not input_audio.exists():
        print(f"Error: Audio file not found: {input_audio}")
        sys.exit(1)
    
    # Create output filename
    output_audio = input_audio.parent / f"{input_audio.stem}_protected.flac"
    
    print("="*60)
    print("Enkidu Podcast Protection Test")
    print("="*60)
    print(f"Input audio: {input_audio}")
    print(f"Output audio: {output_audio}")
    print()
    
    # Initialize pipeline with pre-trained noise patterns
    noise_path = Path("noise_patterns_librispeech.pt")
    
    if not noise_path.exists():
        print(f"Error: Noise patterns not found at {noise_path}")
        print("Have you trained the model? Run train_librispeech.py first.")
        sys.exit(1)
    
    print("Initializing pipeline...")
    pipeline = EnkiduPipeline()
    print("✓ Pipeline initialized")
    print()
    
    # Load pre-trained noise patterns
    print("Loading pre-trained noise patterns...")
    noise_real, noise_imag = pipeline.load_noise(str(noise_path))
    print()
    
    # Protect the audio
    print("Protecting audio...")
    protected_audio = pipeline.protect_audio_file(
        input_path=str(input_audio),
        output_path=str(output_audio),
        noise_real=noise_real,
        noise_imag=noise_imag
    )
    print()
    
    # Load original audio for comparison
    print("Loading original audio for evaluation...")
    original_audio, sr = torchaudio.load(str(input_audio))
    if original_audio.size(0) > 1:
        original_audio = original_audio.mean(dim=0, keepdim=True)
    if sr != pipeline.sample_rate:
        original_audio = torchaudio.functional.resample(
            original_audio, sr, pipeline.sample_rate
        )
    print()
    
    # Evaluate protection effectiveness
    results = pipeline.evaluate_protection(
        original_audio=original_audio,
        protected_audio=protected_audio
    )
    similarity = results['similarity']
    
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Speaker similarity score: {similarity:.4f}")
    print()
    
    if similarity < 0.5:
        print("✓ EXCELLENT PROTECTION!")
        print("  Score below 0.5 means speaker recognition systems")
        print("  cannot reliably identify the original speaker.")
    elif similarity < 0.7:
        print("✓ GOOD PROTECTION")
        print("  Moderate protection - speaker is partially obscured.")
    else:
        print("⚠ WEAK PROTECTION")
        print("  Score above 0.7 means the speaker is still recognizable.")
    
    print()
    print(f"Original: {input_audio}")
    print(f"Protected: {output_audio}")
    print("="*60)


if __name__ == "__main__":
    main()