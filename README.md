# 🎤 Enkidu Voice Privacy Protection

**Adversarial Audio Protection Against Speaker Recognition Systems**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-red.svg)
![License](https://img.shields.io/badge/License-Educational-green.svg)

*Capstone Project - Python Software Engineer Certification @ nomades advanced technologies (Oct-Dec 2025)*

---

## 🎯 Overview

Enkidu is an adversarial audio protection system that applies imperceptible noise to speech recordings, preventing AI-based speaker identification while maintaining audio quality for human listeners.

**What makes this project unique:**
- 🛡️ **Universal Protection**: Single training session protects any voice
- 🎵 **Quality Preservation**: Imperceptible changes to human ears
- ⚡ **Optimized Training**: 3-5 minutes on Apple Silicon M4 GPU
- 🖥️ **Dual Interface**: Professional CLI API + interactive Streamlit GUI
- 📊 **Complete Pipeline**: From training to deployment

### Project Goals

This certification project demonstrates:
1. **Machine Learning Engineering**: PyTorch pipelines, adversarial training, evaluation metrics
2. **Audio Signal Processing**: STFT/ISTFT transformations, frequency-domain operations
3. **Software Architecture**: Object-oriented design, lazy loading, configuration management
4. **Practical Application**: Real privacy protection tool with professional UX

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [How It Works](#-how-it-works)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage Examples](#-usage-examples)
- [Evaluation Results](#-evaluation-results)
- [Project Architecture](#-project-architecture)
- [Learning Outcomes](#-learning-outcomes)
- [Future Improvements](#-future-improvements)
- [License & Credits](#-license--credits)

---

## ✨ Key Features

### Core Functionality
- ✅ **Universal Adversarial Perturbations**: Trained noise patterns generalize across speakers
- ✅ **Frequency-Domain Protection**: STFT-based modifications preserve perceptual quality
- ✅ **Comprehensive Evaluation**: Cosine similarity metrics with speaker embeddings
- ✅ **Hybrid Architecture**: Native training (MPS) + containerized inference

### User Interfaces
- 🖥️ **Command-Line Pipeline**: Clean Python API for batch processing
- 🌐 **Streamlit GUI**: Interactive web interface with real-time playback
- 📊 **Spectrogram Visualization**: Side-by-side comparison of original vs. protected audio

### Technical Highlights
- 🚀 **MPS Acceleration**: 10-15x speedup on Apple Silicon vs CPU
- 🎓 **Educational Design**: Step-by-step learning with incremental complexity
- 📦 **Modular Architecture**: Separation of training scripts and inference pipeline

---

## 🔬 How It Works

### The Science

1. **Speaker Embeddings**: Neural networks extract high-dimensional voice "fingerprints"
2. **Adversarial Optimization**: Gradient-based training finds noise that maximizes embedding distance
3. **Universal Perturbations**: Single pattern exploits neural network architecture vulnerabilities
4. **Perceptual Masking**: Frequency-domain noise remains imperceptible to humans

### Protection Pipeline
```
Input Audio → STFT → Apply Adversarial Noise → ISTFT → Protected Audio
               ↓         (frequency domain)         ↓
         Spectrogram                         Time Domain
```

### Evaluation Methodology

**Metric**: Cosine similarity between ECAPA-TDNN speaker embeddings

| Similarity Score | Protection Level | Interpretation |
|-----------------|------------------|----------------|
| < 0.5 | 🟢 Excellent | Speaker unrecognizable |
| 0.5 - 0.7 | 🟡 Good | Partially obscured |
| 0.7 - 0.85 | 🟠 Moderate | Weakly protected |
| > 0.85 | 🔴 Weak | Easily recognized |

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/horohoro8/Enkidu-Voice-Privacy.git
cd Enkidu-Voice-Privacy

# Create virtual environment
python -m venv enkidu_env
source enkidu_env/bin/activate  # Windows: enkidu_env\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install torch torchaudio speechbrain streamlit matplotlib numpy

# Verify Enkidu dependency is present
ls dependencies/Enkidu/core/
# Should show: __init__.py and other core modules
```

### Train Noise Patterns
```bash
# Option 1: Real speech data (LibriSpeech) - Recommended
python scripts/train_librispeech.py

# Option 2: Synthetic data (faster demo)
python scripts/train_native.py
```

### Protect Audio (CLI)
```python
from src.enkidu_experiments.enkidu_pipeline import EnkiduPipeline

pipeline = EnkiduPipeline()
noise_real, noise_imag = pipeline.load_noise('models/noise_patterns_librispeech.pt')

pipeline.protect_audio_file(
    'original.wav',
    'protected.wav',
    noise_real,
    noise_imag
)
```

### Launch GUI
```bash
streamlit run gui/enkidu_gui.py
# Navigate to http://localhost:8501
```

---

## 💡 Usage Examples

### Example 1: Protect a Podcast Clip
```bash
# Download audio segment
python scripts/download_podcast.py "https://youtube.com/watch?v=EXAMPLE" --duration 60

# Apply protection
python scripts/test_podcast.py audio_samples/podcast_cropped_60s.wav
```

### Example 2: Batch Processing
```python
from pathlib import Path
from src.enkidu_experiments.enkidu_pipeline import EnkiduPipeline

pipeline = EnkiduPipeline()
noise_real, noise_imag = pipeline.load_noise('models/noise_patterns_librispeech.pt')

# Process directory
for audio_file in Path('audio_samples').glob('*.wav'):
    output = f"protected_{audio_file.name}"
    pipeline.protect_audio_file(str(audio_file), output, noise_real, noise_imag)
```

### Example 3: Custom Configuration
```python
config = {
    'device': 'mps',      # Apple Silicon GPU
    'steps': 50,          # Training iterations
    'alpha': 0.15,        # Perturbation step size
    'noise_level': 0.5    # Noise strength
}

pipeline = EnkiduPipeline(config=config)
```

---

## 📊 Evaluation Results

### LibriSpeech Training Performance

| Metric | Value | Status |
|--------|-------|--------|
| **Similarity Score** | 0.4852 | 🟢 Excellent |
| **Training Time** | ~4 min | ⚡ Fast (M4 MPS) |
| **Audio Quality** | Imperceptible | ✅ Preserved |
| **Dataset** | LibriSpeech dev-clean | 📚 20 samples |

### Domain Adaptation Analysis

| Audio Type | Similarity | Protection | Notes |
|------------|-----------|------------|-------|
| LibriSpeech (clean) | 0.48 | 🟢 Excellent | Training domain |
| Podcast (conversational) | 0.81-0.85 | 🟠 Moderate | Domain shift |
| Synthetic speech | 0.52 | 🟢 Excellent | Controlled test |

**Key Insight**: Protection effectiveness varies across acoustic domains - an important ML generalization lesson documented transparently for educational purposes.

---

## 🏗️ Project Architecture

### File Structure
```
Enkidu-Voice-Privacy/
├── src/
│   ├── enkidu_experiments/
│   │   ├── __init__.py
│   │   └── enkidu_pipeline.py       # Core EnkiduPipeline class
│   └── data_prep/
│       ├── __init__.py
│       └── prepare_librispeech.py   # Data preparation utilities
├── scripts/
│   ├── train_librispeech.py         # LibriSpeech training
│   ├── train_native.py              # Synthetic data training
│   ├── test_podcast.py              # Testing utility
│   ├── download_podcast.py          # Audio download helper
│   ├── experiment_noise_levels.py   # Noise optimization experiments
│   └── compare_protection.py        # Protection comparison
├── gui/
│   └── enkidu_gui.py                # Streamlit web interface
├── dependencies/
│   └── Enkidu/
│       ├── core/                    # Enkidu core module
│       ├── wavdataset/              # Dataset utilities
│       ├── enkidu_cli.py
│       └── test_enkidu.py
├── models/                          # Create this directory
│   └── .gitkeep                     # (trained .pt files not in repo)
├── requirements.txt                 # Production dependencies
├── requirements_native.txt          # Development dependencies
├── .gitignore
└── README.md
```

**Key Directories:**
- **`src/enkidu_experiments/`**: Core pipeline implementation
- **`scripts/`**: Training, testing, and utility scripts
- **`gui/`**: Streamlit web interface
- **`dependencies/Enkidu/`**: External adversarial audio library
- **`models/`**: Directory for trained noise patterns (create locally, `.pt` files not tracked)

**Local directories (not in repo):**
After cloning, you'll need to create these directories locally as you use the tools:
- `audio_samples/` - Test audio files (generated during use)
- `data/` - Dataset downloads (created by training scripts)
- `learning/` - Educational materials (optional)
- `enkidu_native_env/` - Virtual environment (created during setup)
- `*.pt` files in `models/` - Trained noise patterns (generated by training)

### EnkiduPipeline Class Design

**Key Design Patterns:**
```python
class EnkiduPipeline:
    # Lazy loading with @property
    @property
    def speaker_model(self):
        if self._speaker_model is None:
            self._speaker_model = SpeakerRecognition.from_hparams(...)
        return self._speaker_model
    
    # Configuration management
    DEFAULT_CONFIG = {
        'device': 'cpu',
        'steps': 40,
        'alpha': 0.1,
        ...
    }
    
    # Device abstraction (CPU/GPU/MPS)
    # Error handling with clear messages
    # Comprehensive docstrings
```

**Separation of Concerns:**
- **Core package** (`src/enkidu_experiments/`): Pipeline class with OOP design
- **Scripts** (`scripts/`): Training, testing, and utility scripts
- **GUI** (`gui/`): Streamlit web interface
- **Dependencies** (`dependencies/Enkidu/`): External adversarial library
- **Models** (`models/`): Directory for trained noise patterns (files generated locally)

---

## 🎓 Learning Outcomes

### Technical Skills Demonstrated

1. **Machine Learning Engineering**
   - Adversarial training with gradient-based optimization
   - Speaker embedding extraction and similarity metrics
   - Universal perturbation concept vs speaker-specific approaches
   - Domain adaptation challenges and solutions

2. **Audio Signal Processing**
   - Short-Time Fourier Transform (STFT) operations
   - Frequency-domain manipulation with perceptual masking
   - Audio I/O with multiple formats (WAV, FLAC)
   - Resampling and mono/stereo conversion

3. **Python Best Practices**
   - Object-oriented design with lazy loading (`@property`)
   - Configuration management with default overrides
   - Device-agnostic code (CPU/GPU/MPS abstraction)
   - Comprehensive error handling and logging
   - Type hints and docstrings

4. **Software Architecture**
   - Separation of training (scripts) and inference (classes)
   - Modular design with clear interfaces
   - Checkpointing for model persistence
   - Professional API design

### Problem-Solving Approach

- **Systematic debugging**: PyTorch compatibility, import paths, device management
- **Appropriate scoping**: Educational value over complexity (no over-engineering)
- **Transparency**: Openly documenting limitations (domain adaptation)
- **Critical thinking**: Understanding why single-speaker training generalizes

---

## 🔮 Future Improvements

### Multi-Method Integration

1. **V-Cloak: Voice Transformation**
   - Integrate V-Cloak for timbre modification and voice characteristic alteration
   - Complement adversarial perturbations with voice transformation
   - Allow users to choose between Enkidu (noise-based) and V-Cloak (transformation-based)
   - Combine both methods for multi-layered protection

2. **AntiFake: Deepfake Detection**
   - Add deepfake detection capabilities to identify AI-generated voices
   - Dual functionality: protect your voice AND detect fake voices
   - Pre-screen audio before applying protection
   - Provide authenticity verification for suspicious recordings

3. **Additional Protection Methods**
   - Explore Fawkes-Audio for voice cloning protection
   - Research AdvPulse for time-domain perturbations
   - Investigate neural voice camouflage approaches
   - Build unified protection suite with method comparison

### Technical Enhancements

1. **Domain Adaptation**
   - Multi-domain training (LibriSpeech + podcasts + phone calls)
   - Data augmentation (background noise, reverberation, codec simulation)
   - Transfer learning from clean to noisy domains

2. **Real-Time Processing**
   - Streaming audio from microphone input
   - Latency optimization (<100ms)
   - Voice activity detection (VAD)

3. **Enhanced Evaluation**
   - Perceptual quality metrics (PESQ, STOI, MOS)
   - Multiple ASV systems (x-vectors, Resemblyzer)
   - Listening tests with human subjects

### Feature Additions

- Adjustable protection strength slider
- Support for MP3, AAC, OGG formats
- Video audio track processing
- Batch processing GUI
- REST API for cloud deployment
- Docker containerization

---

## 📚 Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **ML Framework** | PyTorch 2.9.1 | Tensor operations, gradient computation |
| **Audio Processing** | torchaudio 2.9.1 | I/O, STFT, resampling |
| **Speaker Recognition** | SpeechBrain 0.5.16 | ECAPA-TDNN embeddings |
| **GUI** | Streamlit | Web interface |
| **Visualization** | Matplotlib | Spectrograms |
| **Acceleration** | MPS (Metal) | Apple Silicon GPU |
| **Dataset** | LibriSpeech | Training corpus |

---

## 📄 License & Credits

### Author

**Mark ALLADO**  
Python Software Engineer Certification  
nomades advanced technologies, Genève  
October-December 2025

📧 allado.mark@proton.me  
🐙 [@horohoro8](https://github.com/horohoro8)

### Acknowledgments

- **Enkidu Framework**: Base adversarial audio library ([GitHub](https://github.com/NoobCodeNameless/Enkidu))
- **SpeechBrain**: Pre-trained speaker recognition models ([Website](https://speechbrain.github.io/))
- **LibriSpeech**: Speech corpus by Vassil Panayotov et al.
- **nomades advanced technologies**: Educational supervision and project guidance
- **Course instructors**: Feedback and technical mentorship

### License

This project was developed as part of an educational certification program. It is provided for educational and research purposes.

The code demonstrates Python software engineering skills learned during the certification and may be used as reference material for educational purposes with proper attribution.

---

## 🔗 Related Resources

- [GitHub Repository](https://github.com/horohoro8/Enkidu-Voice-Privacy)
- [nomades advanced technologies](https://nomades.ch/)

---

**⭐ If you found this project interesting, please star the repository!**

*Developed as a Python Software Engineer certification capstone project - December 2025*