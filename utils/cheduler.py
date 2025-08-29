import math
import numpy as np  
class LRScheduler:
    def __init__(self, optimizer, mode='step', step_size=20, gamma=0.5, T_max=100, eta_min=0.000001,
                 warmup_epochs=10, warmup_start_lr=0.0000001, cycles=1, constant_epochs=20,
                 patience=5, factor=0.5, threshold=0.01):

        self.optimizer = optimizer
        self.mode = mode
        self.step_size = step_size
        self.gamma = gamma
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.last_epoch = -1
        
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        
        self.cycles = cycles
        
        self.constant_epochs = constant_epochs
        
        self.patience = patience
        self.factor = factor
        self.threshold = threshold
        self.best_val_loss = float('inf')
        self.bad_epochs = 0
        
        if mode == 'one_cycle':
            self.max_lr = self.base_lr * 10  
        elif mode == 'cyclical':
            self.max_lr = self.base_lr * 5
        
        if mode == 'triangular':
            self.max_lr = self.base_lr * 5
        
    def step(self, val_loss=None):
        self.last_epoch += 1
  
        if self.mode == 'plateau' and val_loss is not None:
            if val_loss < self.best_val_loss * (1 - self.threshold):
                self.best_val_loss = val_loss
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= self.patience:
                    self.optimizer.lr = max(self.optimizer.lr * self.factor, self.eta_min)
                    self.bad_epochs = 0
        else:
            lr = self._compute_lr()
            self.optimizer.lr = lr
        
    def _compute_lr(self):
        if self.constant_epochs > 0 and self.last_epoch < self.constant_epochs:
            return self.base_lr
            
        if self.warmup_epochs > 0 and self.last_epoch < self.constant_epochs + self.warmup_epochs:
            adjusted_epoch = self.last_epoch - self.constant_epochs
            return self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * (adjusted_epoch / self.warmup_epochs)
        
        adjusted_epoch = self.last_epoch - self.constant_epochs - (self.warmup_epochs if self.warmup_epochs > 0 else 0)
        
        if self.mode == 'step':
            return self.base_lr * (self.gamma ** (adjusted_epoch // self.step_size))
        
        elif self.mode == 'cosine':
            if adjusted_epoch >= self.T_max:
                return self.eta_min
            return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * adjusted_epoch / self.T_max))
        
        elif self.mode == 'exp':
            return self.base_lr * (self.gamma ** adjusted_epoch)
        
        elif self.mode == 'cyclical':
            cycle_size = self.T_max // self.cycles
            current_cycle = min(adjusted_epoch // cycle_size, self.cycles - 1)
            current_epoch_in_cycle = adjusted_epoch - current_cycle * cycle_size
            
            cosine_value = np.cos(np.pi * current_epoch_in_cycle / cycle_size)
            return self.eta_min + 0.5 * (self.max_lr - self.eta_min) * (1 + cosine_value)
            
        elif self.mode == 'one_cycle':
            first_phase = self.T_max // 2  
            if adjusted_epoch < first_phase:
                progress = adjusted_epoch / first_phase
                return self.base_lr + (self.max_lr - self.base_lr) * progress
            else:
                progress = (adjusted_epoch - first_phase) / (self.T_max - first_phase)
                return self.eta_min + 0.5 * (self.max_lr - self.eta_min) * (1 + math.cos(math.pi * progress))
                
        elif self.mode == 'triangular':
            cycle_size = self.T_max // self.cycles
            cycle = 1 + adjusted_epoch // (2 * cycle_size)
            x = np.abs(adjusted_epoch / cycle_size - 2 * (cycle - 1) - 1)
            scale_factor = 1.0
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * scale_factor
            
        else:  
            return self.base_lr