# Nomades project

![nomades logo](https://nomades.ch/wp-content/themes/nomades_23/assets/imgs/logo-nomades.png)

## PPL_2025_1012
## Mark Allado
## allado.mark@proton.me

## Description

# 🎤 Enkidu Voice Privacy Protection

**Adversarial Audio Protection Against Speaker Recognition Systems**

*Python Certification Project - October-December 2025*

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Project Architecture](#project-architecture)
- [Evaluation Results](#evaluation-results)
- [Limitations & Learning](#limitations--learning)
- [Technical Stack](#technical-stack)
- [Development Journey](#development-journey)
- [Future Improvements](#future-improvements)
- [License & Credits](#license--credits)

---

## Overview

Enkidu is a voice privacy protection system that applies imperceptible adversarial noise to speech recordings, preventing AI-based speaker recognition while maintaining audio quality for human listeners.

This project demonstrates advanced Python concepts including:
- Machine learning pipeline design with PyTorch
- Audio signal processing with torchaudio
- GUI development with Streamlit
- Integration with pre-trained neural networks (SpeechBrain)
- Professional software architecture patterns

### 🎯 Project Goals

1. **Privacy Protection**: Protect speaker identity from Automatic Speaker Verification (ASV) systems
2. **Audio Quality**: Maintain intelligibility and natural sound for human listeners
3. **Educational Value**: Showcase Python software engineering best practices
4. **Practical Application**: Provide a working tool with both CLI and GUI interfaces

---

## Key Features

### ✨ Core Functionality

- **Universal Adversarial Perturbations**: Noise patterns trained on multiple speakers that generalize to new voices
- **Frequency-Domain Protection**: Applies imperceptible modifications in the spectral domain using STFT
- **Quality Preservation**: Maintains audio intelligibility while fooling speaker recognition systems
- **Comprehensive Evaluation**: Measures protection effectiveness using cosine similarity of speaker embeddings

### 🛠️ User Interfaces

- **Command-Line Pipeline**: Professional Python API for batch processing
- **Streamlit GUI**: Interactive web interface with real-time audio playback and spectrogram visualization
- **Modular Design**: Separate training and inference pipelines following ML deployment best practices

### 📊 Visualization & Analysis

- **Spectrogram Comparison**: Side-by-side visualization of original vs. protected audio
- **Protection Metrics**: Quantitative evaluation with similarity scores and protection levels
- **Real-Time Feedback**: Progress indicators and detailed logging throughout processing

---

## How It Works

### The Science Behind Enkidu

1. **Speaker Embeddings**: Neural networks create high-dimensional "fingerprints" from voice recordings
2. **Adversarial Optimization**: Training process learns noise patterns that maximize distance between embeddings
3. **Universal Perturbations**: Single noise pattern works across different speakers by exploiting neural network vulnerabilities
4. **Frequency Masking**: Noise applied in frequency domain with perceptual masking to remain imperceptible

### Protection Pipeline

```
Original Audio → STFT → Apply Adversarial Noise → ISTFT → Protected Audio
                  ↓                                           ↓
          Frequency Domain                            Time Domain
```

### Evaluation Methodology

- **Metric**: Cosine similarity between speaker embeddings (0 = completely different, 1 = identical)
- **Protection Levels**:
  - **Excellent**: Similarity < 0.5 (unrecognizable)
  - **Good**: Similarity 0.5-0.7 (partially obscured)
  - **Moderate**: Similarity 0.7-0.85 (weakly protected)
  - **Weak**: Similarity > 0.85 (easily recognized)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- macOS (for MPS acceleration) or Linux/Windows with CPU/CUDA support
- 4GB+ RAM recommended

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/enkidu-voice-privacy.git
cd enkidu-voice-privacy
```

### Step 2: Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv enkidu_env

# Activate environment
# On macOS/Linux:
source enkidu_env/bin/activate
# On Windows:
enkidu_env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install speechbrain streamlit matplotlib numpy
```

### Step 4: Set Up Enkidu Dependency

```bash
# Create dependencies directory
mkdir -p dependencies

# Clone Enkidu repository
cd dependencies
git clone https://github.com/your-enkidu-fork/Enkidu.git
cd ..
```

---

## Quick Start

### Train Your Noise Patterns

**Option 1: Using LibriSpeech Dataset (Recommended)**

```bash
python train_librispeech.py
```

This downloads the LibriSpeech dev-clean subset (~350MB) and trains universal noise patterns on real speech data.

**Option 2: Using Synthetic Data (Faster)**

```bash
python train_native.py
```

Creates synthetic audio samples for quick training demonstration.

### Protect an Audio File (CLI)

```python
from src.enkidu_experiments.enkidu_pipeline import EnkiduPipeline

# Initialize pipeline
pipeline = EnkiduPipeline()

# Load trained noise patterns
noise_real, noise_imag = pipeline.load_noise('noise_patterns_librispeech.pt')

# Protect audio file
pipeline.protect_audio_file(
    input_path='original.wav',
    output_path='protected.wav',
    noise_real=noise_real,
    noise_imag=noise_imag
)
```

### Launch GUI

```bash
streamlit run enkidu_gui.py
```

Navigate to `http://localhost:8501` in your browser.

---

## Usage Examples

### Example 1: Protect a Podcast Clip

```bash
# Download a podcast segment
python download_podcast.py "https://youtube.com/watch?v=example" --duration 60

# Test protection
python test_podcast.py audio_samples/podcast_cropped_60s.wav
```

### Example 2: Batch Processing

```python
from pathlib import Path
from src.enkidu_experiments.enkidu_pipeline import EnkiduPipeline

pipeline = EnkiduPipeline()
noise_real, noise_imag = pipeline.load_noise('noise_patterns_librispeech.pt')

# Process all files in a directory
input_dir = Path('audio_input')
output_dir = Path('audio_protected')

for audio_file in input_dir.glob('*.wav'):
    output_file = output_dir / f"protected_{audio_file.name}"
    pipeline.protect_audio_file(
        str(audio_file),
        str(output_file),
        noise_real,
        noise_imag
    )
```

### Example 3: Custom Configuration

```python
# Configure pipeline with custom parameters
config = {
    'device': 'mps',  # Use Apple Silicon GPU
    'steps': 50,      # More training iterations
    'alpha': 0.15,    # Larger perturbation steps
    'noise_level': 0.5  # Stronger noise
}

pipeline = EnkiduPipeline(config=config)
```

---

## Project Architecture

### File Structure

```
enkidu-voice-privacy/
├── src/
│   └── enkidu_experiments/
│       └── enkidu_pipeline.py      # Core pipeline class
├── dependencies/
│   └── Enkidu/                     # External adversarial audio library
├── models/
│   └── noise_patterns_librispeech.pt  # Trained noise patterns
├── train_librispeech.py            # Training script (real data)
├── train_native.py                 # Training script (synthetic data)
├── enkidu_gui.py                   # Streamlit web interface
├── test_podcast.py                 # CLI testing utility
├── download_podcast.py             # Audio download helper
└── README.md
```

### Class Design: EnkiduPipeline

The `EnkiduPipeline` class follows object-oriented best practices:

**Key Design Patterns:**
- **Lazy Loading**: Models loaded only when first accessed via `@property` decorators
- **Configuration Management**: Default config with override capability
- **Device Abstraction**: Transparent handling of CPU/GPU/MPS devices
- **Error Handling**: Defensive programming with clear error messages

**Main Methods:**
- `load_noise()`: Load pre-trained adversarial patterns
- `protect_audio_file()`: Complete audio protection pipeline
- `evaluate_protection()`: Quantify protection effectiveness

```python
# Lazy loading example
@property
def speaker_model(self):
    """Load speaker recognition model only when needed"""
    if self._speaker_model is None:
        self._speaker_model = SpeakerRecognition.from_hparams(...)
    return self._speaker_model
```

---

## Evaluation Results

### LibriSpeech Training Results

**Training Configuration:**
- Dataset: LibriSpeech dev-clean subset
- Samples: 20 audio clips from single speaker
- Device: Apple M4 (MPS acceleration)
- Training Time: ~4 minutes

**Protection Performance:**
- Similarity Score: **0.4852**
- Protection Level: **EXCELLENT** ✅
- Audio Quality: Imperceptible to human listeners

### Test Results on Different Audio Types

| Audio Type | Similarity Score | Protection Level | Notes |
|------------|------------------|------------------|-------|
| LibriSpeech (training domain) | 0.48 | Excellent ✅ | Clean read speech |
| Podcast audio | 0.81-0.85 | Moderate ⚠️ | Domain shift challenges |
| Synthetic speech | 0.52 | Excellent ✅ | Controlled testing |

---

## Limitations & Learning

### Domain Adaptation Challenges

**Key Finding**: Protection trained on clean LibriSpeech data shows reduced effectiveness on real-world conversational audio.

**Why This Happens:**
- **Acoustic Mismatch**: LibriSpeech contains clean, read speech while podcasts have natural conversation
- **Background Noise**: Real-world recordings include music, room acoustics, and environmental sounds
- **Speaking Style**: Read vs. spontaneous speech have different prosodic patterns

**Educational Value**: This demonstrates the importance of domain adaptation in machine learning systems - a critical concept in real-world ML deployment.

### Transparency in Presentation

For the certification presentation, this limitation is discussed openly as:
1. A valuable learning experience about ML generalization
2. Evidence of critical thinking and scientific honesty
3. An opportunity to propose future improvements (domain adaptation, data augmentation)

---

## Technical Stack

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.9.1 | Neural network operations & tensor computation |
| torchaudio | 2.9.1 | Audio I/O and signal processing |
| SpeechBrain | 0.5.16 | Pre-trained speaker recognition models |
| Streamlit | Latest | Web-based GUI framework |
| Matplotlib | Latest | Spectrogram visualization |
| NumPy | Latest | Numerical operations |

### Development Environment

- **Primary**: macOS with Apple Silicon (M4)
- **Acceleration**: MPS (Metal Performance Shaders) for GPU
- **Python**: 3.10+
- **IDE**: VS Code with Python extensions

---

## Development Journey

### Learning Highlights

1. **Adversarial ML Concepts**
   - Learned that adversarial perturbations are largely universal
   - Understood gradient-based optimization for noise generation
   - Discovered perceptual masking techniques

2. **Audio Signal Processing**
   - Mastered STFT (Short-Time Fourier Transform) for frequency analysis
   - Learned complex number mathematics for phase/magnitude manipulation
   - Understood sampling rates and audio normalization

3. **Python Best Practices**
   - Implemented lazy loading with `@property` decorators
   - Used dictionary comprehensions with filtering
   - Applied defensive programming patterns
   - Created modular, testable code architecture

4. **ML Pipeline Design**
   - Separated training (procedural scripts) from inference (OOP classes)
   - Implemented proper device management (CPU/GPU/MPS)
   - Used checkpointing for model persistence
   - Applied evaluation metrics appropriate for the task

### Code Quality Practices

- **Type Hints**: Clear function signatures with expected types
- **Documentation**: Comprehensive docstrings for all public methods
- **Error Handling**: Graceful degradation with helpful error messages
- **Logging**: Informative progress indicators with emoji markers
- **Configuration**: Externalized hyperparameters in config dictionaries

---

## Future Improvements

### Technical Enhancements

1. **Domain Adaptation**
   - Train on diverse audio sources (podcasts, phone calls, recordings)
   - Implement data augmentation (noise injection, reverberation)
   - Fine-tune on target domain after initial training

2. **Real-Time Processing**
   - Optimize for streaming audio (microphone input)
   - Reduce latency with efficient algorithms
   - Implement voice activity detection

3. **Enhanced Evaluation**
   - Add perceptual quality metrics (PESQ, STOI)
   - Test against multiple ASV systems
   - Implement A/B listening tests

### Feature Additions

1. **Adaptive Protection Levels**
   - User-adjustable protection strength slider
   - Automatic quality-privacy trade-off optimization
   - Speaker-specific fine-tuning options

2. **Extended Format Support**
   - Support for MP3, AAC, OGG formats
   - Video file audio track extraction
   - Real-time streaming from URLs

3. **Cloud Deployment**
   - Containerization with Docker
   - REST API for remote processing
   - Web-based frontend (React/Vue.js)

---

## License & Credits

### License

This project is developed as part of the Python Software Engineer certification at nomades advanced technologies. It is provided for educational and research purposes.

### Credits

**Developed by**: Mark RALLIE ALLADO  
**Certification**: Python Programming Language - Oct-Dec 2025  
**Institution**: nomades advanced technologies, Zürich

**External Libraries:**
- [Enkidu](https://github.com/voiceprivacy/Enkidu) - Adversarial audio protection framework
- [SpeechBrain](https://speechbrain.github.io/) - Speaker recognition models
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [LibriSpeech](https://www.openslr.org/12/) - Speech corpus dataset

**Special Thanks:**
- Course instructors for project guidance and feedback
- SpeechBrain team for ECAPA-TDNN pre-trained models
- Enkidu authors for foundational adversarial audio research

---

## Contact & Contribution

### Questions or Feedback?

If you have questions about this project or want to discuss voice privacy protection:

- 📧 Email: [your-email@example.com]
- 💼 LinkedIn: [Your LinkedIn Profile]
- 🐙 GitHub: [@your-username](https://github.com/your-username)

### Contribution Guidelines

This is an educational certification project, but suggestions and improvements are welcome! Please open an issue or pull request if you:

- Find bugs or documentation errors
- Have ideas for improvements
- Want to extend the functionality
- Can help with domain adaptation challenges

---

## Presentation Information

**Presentation Date**: [To be determined]  
**Duration**: 30 minutes (10 min presentation + 20 min code review)  
**Audience**: Certification jury and fellow students

**Key Presentation Points:**
1. Application usage demonstration (GUI + CLI)
2. Competitive analysis vs. existing solutions
3. Target audience identification
4. MVP functionality showcase
5. Architecture visualization
6. Challenges encountered and solutions

---

## Appendix: Technical Details

### Cosine Similarity Explained

Speaker embeddings are high-dimensional vectors (typically 192-512 dimensions) that represent voice characteristics. Cosine similarity measures the angle between two vectors:

```
similarity = (A · B) / (||A|| × ||B||)
```

**Why Cosine Similarity?**
- Invariant to amplitude scaling (volume doesn't affect similarity)
- Measures angular distance in high-dimensional space
- Standard metric for speaker verification tasks
- Range: -1 (opposite) to +1 (identical)

### STFT Parameters

The Short-Time Fourier Transform converts audio from time domain to frequency domain:

- **Window Size**: 1024 samples (~64ms at 16kHz)
- **Hop Length**: 512 samples (50% overlap)
- **Window Function**: Hann window for smooth transitions
- **FFT Size**: 1024 points

These parameters balance time-frequency resolution for speech signals.

---

**⭐ If you found this project interesting, please star the repository!**

*Last updated: December 2025*
