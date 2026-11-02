'''Modulo: audio.py'''
import torch
from perception.sensor_encoder import SensorEncoder

class AudioEncoder:
    def __init__(self, frequency_bands: int = 8, config=None):
        options = (config or {}).get("perception", {}).get("audio", {})
        self.frequency_bands = int(options.get("frequency_bands", frequency_bands))
        self.encoder = SensorEncoder(
            max_rate=float(options.get("max_rate", 120.0)),
            dt=float(options.get("dt", 0.01)),
        )

    def encode_spectrum(self, spectrum_tensor: torch.Tensor) -> torch.Tensor:
        """
        Converte bandas de frequência sonora em trens de spikes temporais.
        """
        if spectrum_tensor.numel() > self.frequency_bands:
            features = spectrum_tensor.flatten()[:self.frequency_bands]
        else:
            features = torch.nn.functional.pad(spectrum_tensor.flatten(), (0, self.frequency_bands - spectrum_tensor.numel()))
            
        return self.encoder.encode_rate(features)
