'''Modulo: memory/manager.py'''
import os
import torch
from core.spike_engine import SpikeEngine

class MemoryManager:
    def __init__(self, engine: SpikeEngine, save_dir: str = "checkpoints/"):
        """
        Gerenciador de Memória Cognitiva.
        Responsável por armazenamento episódico (curto prazo) e consolidação estrutural (longo prazo).
        """
        self.engine = engine
        self.save_dir = save_dir
        
        # Cria o diretório de memórias se não existir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # Memória de Curto Prazo (Working Memory)
        self.episodic_buffer = []
        self.max_buffer_size = 1000

    # =====================================================================
    # 1. MEMÓRIA EPISÓDICA E ASSOCIAÇÃO (Curto Prazo)
    # =====================================================================
    def store_experience(self, sensory_state: list[float], action: int, reward: float):
        """
        Armazena a percepção, decisão e consequência imediata.
        Útil para reprodução de memórias (Experience Replay) no futuro.
        """
        experience = {
            "state": sensory_state,
            "action": action,
            "reward": reward
        }
        
        self.episodic_buffer.append(experience)
        
        # Esquece as memórias mais antigas se o buffer encher (limite de atenção)
        if len(self.episodic_buffer) > self.max_buffer_size:
            self.episodic_buffer.pop(0)

    def clear_short_term_memory(self):
        """Limpa a memória de trabalho (ex: após mudança drástica de cenário)."""
        self.episodic_buffer.clear()

    # =====================================================================
    # 2. CONSOLIDAÇÃO E RECUPERAÇÃO (Longo Prazo / Engramas)
    # =====================================================================
    def consolidate_memory(self, memory_name: str = "latest_brain_engram.pt"):
        """
        Dorme/Consolida: Extrai as matrizes esparsas atuais do motor (SpikeEngine)
        e salva de forma otimizada no disco.
        """
        filepath = os.path.join(self.save_dir, memory_name)
        
        # Dicionário para guardar a topologia de todas as camadas
        engram_data = {}
        
        for idx, synapse_layer in enumerate(self.engine.synapses):
            # Extrai apenas índices e valores (ignora os zeros da rede esparsa)
            engram_data[f"layer_{idx}"] = {
                'indices': synapse_layer.sparse_weights.indices(),
                'values': synapse_layer.sparse_weights.values(),
                'size': synapse_layer.sparse_weights.size()
            }
            
        torch.save(engram_data, filepath)
        print(f"[Memória] Engrama consolidado com sucesso em: {filepath}")

    def retrieve_memory(self, memory_name: str = "latest_brain_engram.pt"):
        """
        Acorda/Recupera: Carrega os pesos de um engrama salvo e injeta 
        diretamente nas sinapses do motor (SpikeEngine).
        """
        filepath = os.path.join(self.save_dir, memory_name)
        
        if not os.path.exists(filepath):
            print(f"[Memória] Engrama {memory_name} não encontrado. Iniciando com cérebro em branco.")
            return False
            
        engram_data = torch.load(filepath)
        
        for idx, synapse_layer in enumerate(self.engine.synapses):
            layer_data = engram_data[f"layer_{idx}"]
            
            # Reconstrói a rede esparsa exatamente como foi salva
            restored_weights = torch.sparse_coo_tensor(
                layer_data['indices'], 
                layer_data['values'], 
                size=layer_data['size']
            ).coalesce()
            
            # Injeta na rede do agente
            synapse_layer.sparse_weights = restored_weights
            
        print(f"[Memória] Engrama {memory_name} recuperado e associado à rede com sucesso!")
        return True