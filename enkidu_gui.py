"""
Enkidu Voice Privacy Protection - GUI
A Streamlit interface for protecting audio files from speaker recognition
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
import torchaudio
import torch
import matplotlib.pyplot as plt
import numpy as np

# Import your actual EnkiduPipeline
# Adjust this import path based on where you run the script from
try:
    from src.enkidu_experiments.enkidu_pipeline import EnkiduPipeline
except ImportError:
    st.error("⚠️ Could not import EnkiduPipeline. Make sure you're running from the project root directory.")
    st.stop()


def create_spectrogram(audio_tensor, sample_rate=16000, title="Spectrogram"):
    """
    Create a spectrogram visualization from audio tensor
    
    Args:
        audio_tensor: Audio tensor (1, samples) or (samples,)
        sample_rate: Sample rate of audio
        title: Title for the plot
    
    Returns:
        matplotlib figure
    """
    # Ensure audio is 1D
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.squeeze()
    
    # Convert to numpy
    audio_np = audio_tensor.cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Create spectrogram using matplotlib
    spectrum, freqs, times, im = ax.specgram(
        audio_np,
        Fs=sample_rate,
        NFFT=1024,
        noverlap=512,
        cmap='viridis',
        scale='dB'  # Use decibel scale for intensity
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Intensity (dB)', rotation=270, labelpad=15)
    
    # Labels and title
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    
    # Set y-axis to log scale and set limits to 20 Hz - 20 kHz
    ax.set_yscale('log')
    ax.set_ylim(20, 20000)
    
    plt.tight_layout()
    return fig


# Page configuration
st.set_page_config(
    page_title="Enkidu Voice Privacy",
    page_icon="🎤",
    layout="wide"
)

# Title and description
st.title("🎤 Enkidu Voice Privacy Protection")
st.markdown("""
This application protects your audio from speaker recognition systems 
while maintaining audio quality for legitimate use cases.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Noise pattern path
noise_path = st.sidebar.text_input(
    "Noise Pattern Path",
    value="models/adversarial_noise.pt",
    help="Path to your trained noise pattern file"
)

# Device selection
device = st.sidebar.selectbox(
    "Device",
    options=["cpu", "mps", "cuda"],
    index=1 if st.sidebar.checkbox("Use MPS (Apple GPU)", value=True) else 0,
    help="Processing device (use 'mps' for Apple Silicon GPU)"
)

# Advanced settings (collapsed by default)
with st.sidebar.expander("Advanced Settings"):
    steps = st.number_input("Steps", value=40, min_value=1, max_value=100)
    alpha = st.number_input("Alpha", value=0.1, min_value=0.0, max_value=1.0, step=0.01)
    mask_ratio = st.number_input("Mask Ratio", value=0.3, min_value=0.0, max_value=1.0, step=0.1)
    frame_length = st.number_input("Frame Length", value=30, min_value=1, max_value=100)
    noise_level = st.number_input("Noise Level", value=0.4, min_value=0.0, max_value=1.0, step=0.1)

# Initialize pipeline in session state (only once)
if 'pipeline' not in st.session_state:
    try:
        with st.spinner("Loading EnkiduPipeline..."):
            # Create config dict with custom values
            config = {
                'device': device,
                'steps': steps,
                'alpha': alpha,
                'mask_ratio': mask_ratio,
                'frame_length': frame_length,
                'noise_level': noise_level,
            }
            st.session_state['pipeline'] = EnkiduPipeline(config=config)
        st.sidebar.success("✅ Pipeline loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading pipeline: {str(e)}")
        import traceback
        st.sidebar.code(traceback.format_exc())

# Load noise patterns
if 'pipeline' in st.session_state and 'noise_patterns' not in st.session_state:
    if Path(noise_path).exists():
        try:
            with st.spinner("Loading noise patterns..."):
                pipeline = st.session_state['pipeline']
                noise_real, noise_imag = pipeline.load_noise(noise_path)
                st.session_state['noise_patterns'] = (noise_real, noise_imag)
            st.sidebar.success("✅ Noise patterns loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading noise: {str(e)}")
    else:
        st.sidebar.warning(f"⚠️ Noise file not found: {noise_path}")

# File uploader
st.header("1. Upload Audio File")
uploaded_file = st.file_uploader(
    "Choose an audio file (WAV format recommended)",
    type=["wav", "mp3", "flac", "ogg"],
    help="Upload the audio file you want to protect"
)

# Display file information if uploaded
if uploaded_file is not None:
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    # Show basic file info
    st.subheader("File Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Filename", uploaded_file.name)
    
    with col2:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.metric("File Size", f"{file_size_mb:.2f} MB")
    
    with col3:
        st.metric("File Type", uploaded_file.type)
    
    # Audio playback section
    st.subheader("Original Audio")
    st.audio(uploaded_file, format=uploaded_file.type)
    
    st.info("👇 Click the button below to apply protection")
else:
    st.info("👆 Please upload an audio file to get started")

# Protection section
st.header("2. Apply Protection")

if uploaded_file is not None and 'pipeline' in st.session_state and 'noise_patterns' in st.session_state:
    # Protection button
    if st.button("🛡️ Protect Audio", type="primary"):
        with st.spinner("Applying protection... This may take a few moments"):
            try:
                # Create temporary files for processing
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_input:
                    tmp_input.write(uploaded_file.getvalue())
                    tmp_input_path = tmp_input.name
                
                # Create output path
                tmp_output_path = tmp_input_path.replace(".wav", "_protected.wav")
                
                # Get noise patterns and pipeline
                pipeline = st.session_state['pipeline']
                noise_real, noise_imag = st.session_state['noise_patterns']
                
                # Apply protection using EnkiduPipeline
                protected_audio = pipeline.protect_audio_file(
                    tmp_input_path, 
                    tmp_output_path,
                    noise_real,
                    noise_imag
                )
                
                # Load original audio for evaluation
                original_audio, _ = torchaudio.load(tmp_input_path)
                if original_audio.size(0) > 1:
                    original_audio = original_audio.mean(dim=0, keepdim=True)
                
                # Evaluate protection
                evaluation = pipeline.evaluate_protection(original_audio, protected_audio)
                
                # Store results in session state
                with open(tmp_output_path, "rb") as f:
                    st.session_state['protected_audio'] = f.read()
                    st.session_state['protected_filename'] = uploaded_file.name
                    st.session_state['evaluation'] = evaluation
                    # Store audio tensors for spectrograms
                    st.session_state['original_audio_tensor'] = original_audio.cpu()
                    st.session_state['protected_audio_tensor'] = protected_audio.cpu()
                    st.session_state['sample_rate'] = pipeline.sample_rate
                
                # Clean up temporary files
                os.unlink(tmp_input_path)
                if os.path.exists(tmp_output_path):
                    os.unlink(tmp_output_path)
                
                st.success("✅ Protection applied successfully!")
                
            except Exception as e:
                st.error(f"❌ Error during protection: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display protected audio if it exists
    if 'protected_audio' in st.session_state:
        st.subheader("Protected Audio")
        
        # Show evaluation results
        if 'evaluation' in st.session_state:
            eval_data = st.session_state['evaluation']
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Protection Level",
                    eval_data['protection_level'],
                    help="How well the audio is protected"
                )
            
            with col2:
                st.metric(
                    "Similarity Score",
                    f"{eval_data['similarity']:.4f}",
                    help="Lower is better (< 0.5 = excellent)"
                )
        
        # Audio player
        st.audio(st.session_state['protected_audio'], format="audio/wav")
        
        # Download button for protected audio
        st.download_button(
            label="📥 Download Protected Audio",
            data=st.session_state['protected_audio'],
            file_name=f"protected_{st.session_state['protected_filename']}",
            mime="audio/wav"
        )
        
elif uploaded_file is not None and 'pipeline' not in st.session_state:
    st.warning("⚠️ Pipeline not loaded. Check the configuration in the sidebar.")
elif uploaded_file is not None and 'noise_patterns' not in st.session_state:
    st.warning("⚠️ Noise patterns not loaded. Check the noise path in the sidebar.")
else:
    st.write("*Upload an audio file first*")

st.header("3. Results")

# Display spectrograms if available
if 'original_audio_tensor' in st.session_state and 'protected_audio_tensor' in st.session_state:
    st.subheader("Spectrogram Comparison")
    
    # Create two columns for side-by-side spectrograms
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Audio**")
        fig_original = create_spectrogram(
            st.session_state['original_audio_tensor'],
            st.session_state['sample_rate'],
            "Original Audio Spectrogram"
        )
        st.pyplot(fig_original)
        plt.close(fig_original)
    
    with col2:
        st.write("**Protected Audio**")
        fig_protected = create_spectrogram(
            st.session_state['protected_audio_tensor'],
            st.session_state['sample_rate'],
            "Protected Audio Spectrogram"
        )
        st.pyplot(fig_protected)
        plt.close(fig_protected)
    
    # Add explanation
    st.info("""
    **Understanding the Spectrograms:**
    - The spectrograms show frequency content over time
    - Y-axis uses logarithmic scale (20 Hz - 20 kHz, full human hearing range)
    - Protected audio should look visually similar (preserve quality)
    - But subtle differences prevent speaker recognition
    """)
else:
    st.write("*Spectrograms will appear here after protecting an audio file*")