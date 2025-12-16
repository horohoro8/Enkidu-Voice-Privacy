#!/usr/bin/env python3
"""
Download and crop audio from YouTube for testing the Enkidu pipeline.

Usage:
    python download_podcast.py <youtube_url> [--duration SECONDS] [--output OUTPUT_DIR]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def download_audio(url: str, output_dir: Path) -> Path:
    """
    Download audio from YouTube URL using yt-dlp.
    
    Args:
        url: YouTube URL
        output_dir: Directory to save the audio file
        
    Returns:
        Path to the downloaded audio file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output template for yt-dlp
    output_template = str(output_dir / "%(title)s.%(ext)s")
    
    print(f"Downloading audio from: {url}")
    print(f"Saving to: {output_dir}")
    
    # yt-dlp command to download best audio
    cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format", "wav",  # Convert to WAV
        "--audio-quality", "0",  # Best quality
        "-o", output_template,
        url
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        # Find the downloaded file
        wav_files = list(output_dir.glob("*.wav"))
        if not wav_files:
            raise FileNotFoundError("No WAV file found after download")
        
        return wav_files[0]
        
    except subprocess.CalledProcessError as e:
        print(f"Error downloading audio: {e.stderr}", file=sys.stderr)
        raise


def crop_audio(input_file: Path, duration: int, output_dir: Path) -> Path:
    """
    Crop audio file to specified duration using ffmpeg.
    
    Args:
        input_file: Path to input audio file
        duration: Duration in seconds to crop to
        output_dir: Directory to save the cropped audio
        
    Returns:
        Path to the cropped audio file
    """
    output_file = output_dir / f"{input_file.stem}_cropped_{duration}s.wav"
    
    print(f"\nCropping audio to {duration} seconds...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    # ffmpeg command to crop audio
    cmd = [
        "ffmpeg",
        "-i", str(input_file),
        "-t", str(duration),  # Duration
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", "16000",  # 16kHz sample rate (standard for speech)
        "-ac", "1",  # Mono
        "-y",  # Overwrite output file
        str(output_file)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Audio cropped successfully: {output_file}")
        return output_file
        
    except subprocess.CalledProcessError as e:
        print(f"Error cropping audio: {e.stderr}", file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download and crop YouTube audio for Enkidu pipeline testing"
    )
    parser.add_argument(
        "url",
        help="YouTube URL to download"
    )
    parser.add_argument(
        "-d", "--duration",
        type=int,
        default=60,
        help="Duration in seconds to crop (default: 60)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("./audio_samples"),
        help="Output directory (default: ./audio_samples)"
    )
    parser.add_argument(
        "--keep-full",
        action="store_true",
        help="Keep the full downloaded audio file"
    )
    
    args = parser.parse_args()
    
    try:
        # Check if yt-dlp is installed
        subprocess.run(["yt-dlp", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: yt-dlp is not installed", file=sys.stderr)
        print("Install with: pip install yt-dlp", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Check if ffmpeg is installed
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg is not installed", file=sys.stderr)
        print("Install with: apt-get install ffmpeg (Linux) or brew install ffmpeg (Mac)", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Download audio
        full_audio = download_audio(args.url, args.output)
        
        # Crop audio
        cropped_audio = crop_audio(full_audio, args.duration, args.output)
        
        # Remove full audio if not keeping
        if not args.keep_full:
            print(f"\nRemoving full audio file: {full_audio}")
            full_audio.unlink()
        
        print("\n" + "="*60)
        print("✓ SUCCESS!")
        print("="*60)
        print(f"Cropped audio ready: {cropped_audio}")
        print(f"Duration: {args.duration} seconds")
        print(f"\nYou can now test this with your Enkidu pipeline:")
        print(f"  python run_pipeline.py {cropped_audio}")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()