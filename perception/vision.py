'''Modulo: vision.py'''
import torch
from perception.sensor_encoder import SensorEncoder

class VisionEncoder:
    def __init__(self, output_features: int = 16):
        self.output_features = output_features
        self.encoder = SensorEncoder(max_rate=150.0)

    def encode_frame(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        """
        Processa um quadro visual bruto e o codifica em um trem de spikes 
        para o córtex visual.
        """
        # Reduzir ou projetar o quadro para a dimensão de características de saída desejada
        if frame_tensor.numel() > self.output_features:
            # Simulação de pooling/downsampling simples
            flattened = frame_tensor.flatten()[:self.output_features]
        else:
            flattened = torch.nn.functional.pad(frame_tensor.flatten(), (0, self.output_features - frame_tensor.numel()))
            
        return self.encoder.encode_rate(flattened)