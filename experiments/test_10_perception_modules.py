'''Modulo: test_10_perception_modules.py'''
import torch
from perception.sensor_encoder import SensorEncoder
from perception.vision import VisionEncoder
from perception.audio import AudioEncoder
from core.neuron import LeakyIntegrateAndFire

def run_perception_modules_test():
    print("Iniciando Teste dos Codificadores de Percepção (NeuroCore)...")
    
    # 1. Testar Codificador de Sensores Genéricos (Rate Coding)
    sensor_encoder = SensorEncoder(max_rate=100.0)
    raw_sensors = torch.tensor([0.2, 0.8, 0.5, 0.0])
    spikes_sensors = sensor_encoder.encode_rate(raw_sensors, dt=0.01)
    print(f"[Sensor Encoder] Spikes gerados a partir de valores contínuos: {spikes_sensors}")
    
    # 2. Testar Codificador Visual (Vision Encoder)
    vision_encoder = VisionEncoder(output_features=8)
    dummy_image_frame = torch.rand(16, 16) # Simulando um quadro de imagem 16x16
    spikes_vision = vision_encoder.encode_frame(dummy_image_frame)
    print(f"[Vision Encoder] Spikes visuais gerados (dimensão 8): {spikes_vision}")
    
    # 3. Testar Codificador Auditivo (Audio Encoder)
    audio_encoder = AudioEncoder(frequency_bands=4)
    dummy_audio_spectrum = torch.tensor([0.1, 0.9, 0.4, 0.6, 0.2]) # Bandas de frequência
    spikes_audio = audio_encoder.encode_spectrum(dummy_audio_spectrum)
    print(f"[Audio Encoder] Spikes auditivos gerados (bandas de frequência): {spikes_audio}")
    
    # 4. Integrar com o Núcleo Neural (LIF)
    lif_layer = LeakyIntegrateAndFire(num_neurons=8)
    # Alimentar o neurônio LIF com os spikes visuais gerados
    spikes_out, mem_out = lif_layer(spikes_vision)
    print(f"[Neural Integration] Spikes propagados na camada LIF: {spikes_out}")
    print(f"[Neural Integration] Potencial de membrana resultante: {mem_out}")
    
    print("Todos os testes dos módulos de percepção foram concluídos com sucesso!")

if __name__ == "__main__":
    run_perception_modules_test()