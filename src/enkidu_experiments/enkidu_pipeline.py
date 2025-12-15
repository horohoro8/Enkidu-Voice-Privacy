import sys
from pathlib import Path #because os.path is old
# needed for Enkidu
import torch
import torchaudio

# GOAL : to be able to import Enkidu we go back to the root of the project
#need to make it more generalize (this is only for my personal)
# ENKIDU_PATH = "/home/student/workspace/nomades_project/dependencies/Enkidu" 
ENKIDU_PATH = Path(__file__).parent.parent.parent / 'dependencies' / 'Enkidu' #but the file should follow the same structure
sys.path.append(str(ENKIDU_PATH)) #add the enkidu path so that I can import it or else it will only have the default
#python prend le premier qu'il prend, il faut mettre en priorite le ENKIDU_PATH
sys.path.insert(0, str(ENKIDU_PATH))
from core import Enkidu
from speechbrain.inference import SpeakerRecognition


class EnkiduPipeline:

    DEFAULT_CONFIG = {
        'device' : 'cpu',
        'sample_rate' : 16000,
        'steps' : 40,
        'alpha' : 0.1,
        'mask_ratio' : 0.3,
        'frame_length' : 120,
        'noise_level' : 0.4,
    }

    def __init__(self, config=None):
        """
        Initialize the pipeline
        
        Args:
            config (dict, optional): Custom configuration. Uses DEFAULT_CONFIG if not provided.
        """
        # If config provided, use it; otherwise use defaults
        if config is None:
            self.config = self.DEFAULT_CONFIG.copy()  # Make a copy! points to the same adress if not copied -> modify one and it modify the default
        else:
            # Start with defaults, then update with custom values
            self.config = self.DEFAULT_CONFIG.copy()
            self.config.update(config)
        
        # Extract commonly used values
        self.device = self.config['device']
        self.sample_rate = self.config['sample_rate']
        
        # Models (load later when needed - "lazy loading")
        self._speaker_model = None
        self._enkidu_model = None
        
        print(f"✓ Pipeline initialized (device: {self.device})")

    def load_noise(self, noise_path):
        """
        Load pre-trained noise patterns from file
            
        Args:
            noise_path (str): Path to noise_patterns.pt file (from Colab)
                
        Returns:
            tuple: (noise_real, noise_imag) tensors
        """
        print(f"\nLoading noise patterns from: {noise_path}")
            
        # Load the checkpoint file
        checkpoint = torch.load(noise_path, map_location=self.device)
            
        # Extract noise patterns and move to device
        noise_real = checkpoint['noise_real'].to(self.device)
        noise_imag = checkpoint['noise_imag'].to(self.device)
            
        print(f"✓ Noise patterns loaded")
        print(f"  Shape: {noise_real.shape}")
            
        # Print training info if available
        if 'metadata' in checkpoint:
            print(f"  Metadata: {checkpoint['metadata']}")
            
        return noise_real, noise_imag

    @property
    def speaker_model(self):
        """
        Load speaker recognition model (lazy loading)
        Only loads when first accessed
        """
        if self._speaker_model is None:
            print("Loading speaker recognition model...")
            self._speaker_model = SpeakerRecognition.from_hparams(
                'speechbrain/spkrec-ecapa-voxceleb',
                run_opts={"device": self.device}
            )
            print("✓ Speaker model loaded")
        return self._speaker_model

    @property
    def enkidu_model(self):
        """
        Initialize Enkidu model (lazy loading)
        Only loads when first accessed
        """
        if self._enkidu_model is None:
            print("Initializing Enkidu model...")
            self._enkidu_model = Enkidu(
                model=self.speaker_model,
                steps=self.config['steps'],
                alpha=self.config['alpha'],
                mask_ratio=self.config['mask_ratio'],
                frame_length=self.config['frame_length'],
                noise_level=self.config['noise_level'],
                device=self.device
            )
            print("✓ Enkidu model initialized")
        return self._enkidu_model
    

    def protect_audio_file(self, input_path, output_path, noise_real, noise_imag):
        """
        Protect a single audio file
        
        Args:
            input_path (str): Path to original audio file
            output_path (str): Where to save protected audio
            noise_real: Real part of noise (from load_noise)
            noise_imag: Imaginary part of noise (from load_noise)
            
        Returns:
            torch.Tensor: Protected audio tensor
        """
        print(f"\nProtecting: {Path(input_path).name}")
        
        # Step 1: Load the audio file
        audio, sr = torchaudio.load(input_path)
        print(f"  Loaded: {audio.shape[1] / sr:.2f} seconds")
        
        # Step 2: Convert to mono if stereo
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
            print(f"  Converted to mono")
        
        # Step 3: Resample to 16kHz if needed
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
            sr = self.sample_rate
            print(f"  Resampled to {sr} Hz")
        
        # Step 4: Move audio to correct device (CPU/GPU)
        audio = audio.to(self.device)
        
        # Step 5: Apply protection using Enkidu
        protected = self.enkidu_model.add_noise(
            audio,
            noise_real,
            noise_imag,
            mask_ratio=self.config.get('mask_ratio', 0.3),
            random_offset=False,
            noise_smooth=True
        )
        print(f"  Protection applied")
        
        # Step 6: Save protected audio
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Create folder if needed
        torchaudio.save(str(output_path), protected.cpu(), sr)
        
        print(f"✓ Saved to: {output_path}")
        
        return protected

    def evaluate_protection(self, original_audio, protected_audio):
        """
        Evaluate how well the protection works
        
        Args:
            original_audio (torch.Tensor): Original audio tensor
            protected_audio (torch.Tensor): Protected audio tensor
            
        Returns:
            dict: Evaluation results with similarity score and protection level
        """
        print("\nEvaluating protection...")
        
        # Get speaker embeddings from both audios
        with torch.no_grad():
            original_emb = self.speaker_model.encode_batch(
                original_audio.to(self.device))
            protected_emb = self.speaker_model.encode_batch(
                protected_audio.to(self.device))
            
            # Calculate similarity (0 = completely different, 1 = identical)
            similarity = torch.nn.functional.cosine_similarity(
                original_emb, protected_emb, dim=-1
            ).item()
        
        # Classify protection level
        if similarity < 0.5:
            level = "EXCELLENT"
            emoji = "🟢"
        elif similarity < 0.7:
            level = "GOOD"
            emoji = "🟡"
        elif similarity < 0.85:
            level = "MODERATE"
            emoji = "🟠"
        else:
            level = "WEAK"
            emoji = "🔴"
        
        print(f"\n{emoji} Protection Level: {level}")
        print(f"  Similarity Score: {similarity:.4f} (lower is better)")
        
        return {
            'similarity': similarity,
            'protection_level': level
        }

# ==============================================================================
# QUICK TEST (run this file directly to test the pipeline)
# ==============================================================================

if __name__ == '__main__':
    print("Testing EnkiduPipeline...")
    print("="*60)
    
    # Create pipeline
    pipeline = EnkiduPipeline()
    
    print("\n✓ Pipeline created successfully!")
    print("\nTo use this pipeline:")
    print("  1. Train noise on Colab and download noise_patterns.pt")
    print("  2. Then use:")
    print("     noise_real, noise_imag = pipeline.load_noise('noise_patterns.pt')")
    print("     pipeline.protect_audio_file('input.flac', 'output.flac', noise_real, noise_imag)")