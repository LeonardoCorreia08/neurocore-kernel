import torch
import torch.nn as nn

class LeakyIntegrateAndFire(nn.Module):
    def __init__(self, num_neurons: int, threshold: float = 0.5, beta: float = 0.9): # <--- Mude o threshold para 0.5
        super().__init__()
        self.num_neurons = num_neurons
        self.threshold = threshold
        self.beta = beta
        self.mem = torch.zeros(num_neurons)
        
        # Estado do potencial de membrana para cada neurônio
        self.mem = torch.zeros(num_neurons)

    def forward(self, x_input: torch.Tensor):
        """
        x_input: Corrente recebida (I) no passo de tempo atual [batch_size, num_neurons]
        Retorna:
            spikes: Tensor binário indicando se o neurônio disparou (1) ou não (0)
            self.mem: Potencial de membrana atualizado
        """
        # Equação de decaimento do potencial e acumulação da entrada
        self.mem = self.beta * self.mem + x_input
        
        # Condição de disparo (spike)
        spikes = (self.mem >= self.threshold).float()
        
        # Reset do potencial após o disparo (subtrai o limiar)
        self.mem = self.mem - spikes * self.threshold
        
        return spikes, self.mem

    def reset_state(self):
        self.mem = torch.zeros(self.num_neurons)