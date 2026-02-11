#!/usr/bin/env python3
"""
Experiment with different noise_level values to find optimal protection.

This tests multiple noise levels and shows which gives best protection
while maintaining reasonable audio quality.
"""

import sys
from pathlib import Path
import torchaudio
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))
from enkidu_experiments.enkidu_pipeline import EnkiduPipeline


def test_noise_level(pipeline, original_audio, noise_real, noise_imag, noise_level, output_dir):
    """Test a specific noise level"""
    
    # Create a temporary config with this noise level
    test_config = pipeline.config.copy()
    test_config['noise_level'] = noise_level
    
    # Apply protection with this noise level
    protected = pipeline.enkidu_model.add_noise(
        original_audio,
        noise_real,
        noise_imag,
        mask_ratio=test_config['mask_ratio'],
        random_offset=False,
        noise_smooth=True
    )
    
    # Save the protected audio
    output_path = output_dir / f"podcast_noise_{noise_level:.2f}.flac"
    torchaudio.save(str(output_path), protected.cpu(), pipeline.sample_rate)
    
    # Evaluate protection
    with torch.no_grad():
        original_emb = pipeline.speaker_model.encode_batch(original_audio)
        protected_emb = pipeline.speaker_model.encode_batch(protected)
        similarity = torch.nn.functional.cosine_similarity(
            original_emb, protected_emb, dim=-1
        ).item()
    
    return similarity, output_path


def main():
    print("="*60)
    print("Noise Level Optimization Experiment")
    print("="*60)
    
    # Input file
    input_file = Path("audio_samples/The AI Race Heats Up： Can ChatGPT Stay on Top？ ｜ EP 167_cropped_300s.wav")
    
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path("audio_samples/experiments")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nInput: {input_file}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = EnkiduPipeline()
    
    # Load noise patterns
    print("Loading noise patterns...")
    noise_real, noise_imag = pipeline.load_noise("noise_patterns_librispeech.pt")
    
    # Load original audio
    print("Loading original audio...")
    original_audio, sr = torchaudio.load(str(input_file))
    if original_audio.size(0) > 1:
        original_audio = original_audio.mean(dim=0, keepdim=True)
    if sr != pipeline.sample_rate:
        original_audio = torchaudio.functional.resample(original_audio, sr, pipeline.sample_rate)
    original_audio = original_audio.to(pipeline.device)
    
    # Test different noise levels
    noise_levels = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    print("\n" + "="*60)
    print("Testing different noise levels...")
    print("="*60)
    print(f"{'Noise Level':<15} {'Similarity':<15} {'Protection':<15} {'File'}")
    print("-"*60)
    
    results = []
    
    for noise_level in noise_levels:
        similarity, output_path = test_noise_level(
            pipeline, original_audio, noise_real, noise_imag, 
            noise_level, output_dir
        )
        
        # Classify protection
        if similarity < 0.5:
            level = "🟢 EXCELLENT"
        elif similarity < 0.7:
            level = "🟡 GOOD"
        elif similarity < 0.85:
            level = "🟠 MODERATE"
        else:
            level = "🔴 WEAK"
        
        print(f"{noise_level:<15.2f} {similarity:<15.4f} {level:<15} {output_path.name}")
        
        results.append({
            'noise_level': noise_level,
            'similarity': similarity,
            'path': output_path
        })
    
    # Find best result
    best = min(results, key=lambda x: x['similarity'])
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print(f"Best noise level: {best['noise_level']:.2f}")
    print(f"Similarity score: {best['similarity']:.4f}")
    print(f"Output file: {best['path']}")
    print()
    
    if best['similarity'] < 0.5:
        print("✓ Excellent! This achieves strong protection.")
    elif best['similarity'] < 0.7:
        print("✓ Good protection achieved.")
    else:
        print("⚠ Even at higher noise levels, protection is limited.")
        print("This suggests the domain mismatch is significant.")
        print("\nConsider:")
        print("• Retraining with podcast data included")
        print("• Using different masking strategies")
        print("• Accepting moderate protection for this use case")
    
    print("\n" + "="*60)
    print("Listen to the files to check audio quality!")
    print(f"Files saved in: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()