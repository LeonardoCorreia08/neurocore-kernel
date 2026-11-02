'''Modulo: hardware_telemetry.py'''
import time
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


def _settings(config):
    if not config:
        return {}
    return config.get("monitoring", config).get("hardware", {})


class HardwareTelemetry:
    def __init__(self, device_index: int = 0, energy_divisor: float = 1000.0, config: dict = None):
        settings = _settings(config)
        self.device_index = settings.get("device_index", device_index)
        self.energy_divisor = settings.get("energy_divisor", energy_divisor)
        self.initialized = False
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
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
            # NVML total-energy API reports millijoules; 1000 mJ equals 1 J.
            energy_joules = (final_energy - self.initial_energy) / self.energy_divisor
        return {"elapsed_seconds": elapsed_time, "energy_joules": energy_joules}

    def _get_energy_uj(self) -> int:
        if not self.initialized:
            return 0
        try:
            # nvmlDeviceGetTotalEnergyConsumption returns millijoules.
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except Exception:
            return 0
