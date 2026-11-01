'''Modulo: hardware_telemetry.py'''
import time
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

class HardwareTelemetry:
    def __init__(self):
        self.initialized = False
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.initialized = True
            except Exception as e:
                print(f"[Telemetry Warning] Não foi possível inicializar NVML: {e}")

    def start_measurement(self):
        """Inicia a contagem de tempo e estado para medição energética."""
        self.start_time = time.time()
        self.initial_energy = self._get_energy_uj()

    def stop_measurement(self) -> dict:
        """Para a medição e retorna o consumo real em Joules e tempo decorrido."""
        end_time = time.time()
        final_energy = self._get_energy_uj()
        
        elapsed_time = end_time - self.start_time
        
        energy_joules = 0.0
        if self.initialized and final_energy >= self.initial_energy:
            # NVML fornece energia em miliJoules ou Joules dependendo da versão, 
            # geralmente _get_energy_uj retorna em microjoules ou mililitros de energia.
            energy_joules = (final_energy - self.initial_energy) / 1000000.0 # Conversão se microjoules
            
        return {
            "elapsed_seconds": elapsed_time,
            "energy_joules": energy_joules
        }

    def _get_energy_uj(self) -> int:
        if not self.initialized:
            return 0
        try:
            # Retorna a energia total consumida pela placa desde o boot (em millijoules)
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except Exception:
            return 0