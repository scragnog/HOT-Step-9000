import os
import torch
import torchaudio
from loguru import logger
from acestep.core.audio.music_vocoder import ADaMoSHiFiGANV1

VOCODER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "checkpoints")

class VocoderService:
    def __init__(self):
        self.vocoders = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def get_available_vocoders(self):
        """Scan checkpoints array for any vocoder folders."""
        vocoders = ["None"]
        if os.path.exists(VOCODER_DIR):
            for item in os.listdir(VOCODER_DIR):
                path = os.path.join(VOCODER_DIR, item)
                if os.path.isdir(path):
                    content = os.listdir(path)
                    if "config.json" in content and any(f.endswith(".safetensors") for f in content):
                        vocoders.append(item)
        return vocoders

    def load_vocoder(self, model_name: str):
        if model_name not in self.vocoders:
            logger.info(f"[VocoderService] Loading vocoder model: {model_name}")
            path = os.path.join(VOCODER_DIR, model_name)
            model = ADaMoSHiFiGANV1.from_pretrained(path, local_files_only=True)
            model = model.to(self.device)
            model.eval()
            self.vocoders[model_name] = model
        return self.vocoders[model_name]

    def apply_vocoder(self, waveform: torch.Tensor, model_name: str, sample_rate: int = 48000) -> torch.Tensor:
        """
        Enhances the waveform by passing it through the vocoder pipeline
        (Waveform -> Mel Spectrogram -> Vocoded Waveform).
        This operates alongside the VAE as a final quality pass.
        """
        if not model_name or model_name.lower() == "none" or model_name not in self.get_available_vocoders():
            return waveform
            
        logger.info(f"[VocoderService] Applying vocoder '{model_name}' to audio")
        model = self.load_vocoder(model_name)
        
        target_sr = 44100
        original_sr = sample_rate
        
        # Ensure correct shape [B, 1, T]
        added_batch = False
        if waveform.dim() == 2:  # [B, T] -> assumes B is channel
            waveform = waveform.unsqueeze(1)
        elif waveform.dim() == 1:  # [T]
            waveform = waveform.unsqueeze(0).unsqueeze(0)
            added_batch = True
            
        was_resampled = False
        if original_sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, original_sr, target_sr)
            was_resampled = True
            
        with torch.no_grad():
            wav_in = waveform.to(self.device).float()
            mel = model.encode(wav_in)
            vocoded_wav = model.decode(mel)
            
        if was_resampled:
            vocoded_wav = torchaudio.functional.resample(vocoded_wav, target_sr, original_sr)
            
        if added_batch:
            vocoded_wav = vocoded_wav.squeeze(0).squeeze(0)
        elif vocoded_wav.dim() == 3 and vocoded_wav.size(0) == 1:
            # Drop the batch dimension [1, C, T] -> [C, T]
            vocoded_wav = vocoded_wav.squeeze(0)
            
        return vocoded_wav.cpu()

vocoder_service = VocoderService()
