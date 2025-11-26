import os
import glob
import shutil

# Adjust these 3 variables as needed
reader_id = 61
librispeech_root = "/home/student/workspace/nomades_project/data/LibriSpeech/test-clean"
output_root = "/home/student/workspace/nomades_project/data/dataset_dir"


def copy_speaker_files():
    # Path to this reader inside LibriSpeech
    reader_path = os.path.join(librispeech_root, str(reader_id))
    print("Reader path:", reader_path)

    # Check folder exists
    if not os.path.exists(reader_path):
        print(f"Reader {reader_id} not found!")
        return

    # Find all .flac files recursively
    voice_files = sorted(
        glob.glob(os.path.join(reader_path, "**", "*.flac"), recursive=True)
    )

    print(f"Found {len(voice_files)} audio files for reader {reader_id}.")

    if not voice_files:
        print("No audio files found, nothing to do.")
        return

    # Create target directory: output_root/speaker_<reader_id>/
    target_dir = os.path.join(output_root, f"speaker_{reader_id}")
    os.makedirs(target_dir, exist_ok=True)
    print("Target directory:", target_dir)

    # Copy and rename files
    for idx, src in enumerate(voice_files):
        new_name = f"{idx:05d}.flac"   # 00000.flac, 00001.flac, ...
        dst = os.path.join(target_dir, new_name)

        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")

    print("\n✅ Done!")
    print(f"All files for reader {reader_id} are now in: {target_dir}")


copy_speaker_files()
