#!/usr/bin/env python3
"""
Compare protection effectiveness between different audio sources.
"""

import sys
from pathlib import Path
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from enkidu_experiments.enkidu_pipeline import EnkiduPipeline


def test_audio_pair(pipeline, original_path, protected_path, label):
    """Test a pair of original/protected audio files"""
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"{'='*60}")
    
    # Load both files
    original, sr1 = torchaudio.load(original_path)
    protected, sr2 = torchaudio.load(protected_path)
    
    # Convert to mono if needed
    if original.size(0) > 1:
        original = original.mean(dim=0, keepdim=True)
    if protected.size(0) > 1:
        protected = protected.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr1 != pipeline.sample_rate:
        original = torchaudio.functional.resample(original, sr1, pipeline.sample_rate)
    if sr2 != pipeline.sample_rate:
        protected = torchaudio.functional.resample(protected, sr2, pipeline.sample_rate)
    
    print(f"Original:  {original_path}")
    print(f"Protected: {protected_path}")
    print(f"Duration:  {original.shape[1] / pipeline.sample_rate:.2f}s")
    
    # Evaluate
    results = pipeline.evaluate_protection(original, protected)
    
    return results


def main():
    print("="*60)
    print("Protection Effectiveness Comparison")
    print("="*60)
    
    # Initialize pipeline
    pipeline = EnkiduPipeline()
    
    # Test cases
    test_cases = [
        {
            'label': 'LibriSpeech Test',
            'original': 'test_original_librispeech.flac',
            'protected': 'test_protected_librispeech.flac'
        },
        {
            'label': 'Podcast Test',
            'original': 'audio_samples/The AI Race Heats Up： Can ChatGPT Stay on Top？ ｜ EP 167_cropped_300s.wav',
            'protected': 'audio_samples/The AI Race Heats Up： Can ChatGPT Stay on Top？ ｜ EP 167_cropped_300s_protected.flac'
        }
    ]
    
    results = []
    
    for test in test_cases:
        original = Path(test['original'])
        protected = Path(test['protected'])
        
        if not original.exists():
            print(f"\n⚠ Skipping {test['label']}: {original} not found")
            continue
        if not protected.exists():
            print(f"\n⚠ Skipping {test['label']}: {protected} not found")
            continue
        
        result = test_audio_pair(
            pipeline,
            str(original),
            str(protected),
            test['label']
        )
        
        results.append({
            'label': test['label'],
            'similarity': result['similarity'],
            'level': result['protection_level']
        })
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    
    for r in results:
        emoji = "🟢" if r['similarity'] < 0.5 else "🟡" if r['similarity'] < 0.7 else "🟠" if r['similarity'] < 0.85 else "🔴"
        print(f"{emoji} {r['label']:20s}: {r['similarity']:.4f} ({r['level']})")
    
    if len(results) >= 2:
        diff = results[1]['similarity'] - results[0]['similarity']
        print(f"\nDifference: {diff:+.4f}")
        print(f"Podcast is {abs(diff):.4f} {'worse' if diff > 0 else 'better'} than LibriSpeech")
        
        print("\n" + "="*60)
        print("ANALYSIS")
        print("="*60)
        print("\nPossible reasons for difference:")
        print("1. Domain mismatch - noise trained on LibriSpeech, tested on podcast")
        print("2. Recording quality differences (studio vs. podcast)")
        print("3. Speaking style differences (read speech vs. conversational)")
        print("4. Compression artifacts from YouTube download")
        print("\nNext steps:")
        print("• Try other podcast segments to see if consistent")
        print("• Consider retraining with mixed data (LibriSpeech + podcasts)")
        print("• Experiment with higher noise_level parameter")


if __name__ == "__main__":
    main()