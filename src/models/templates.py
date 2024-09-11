__author__ = "alvrogd"


import abc
import typing
import uuid

import numpy as np
import torch.nn as nn
import torch.utils.data
import torchinfo

import utils.constants as m_constants
import utils.metrics as m_metrics


class DeepRegressionModel(nn.Module, abc.ABC):
    
    def __init__(self, args: typing.Dict[str, typing.Union[float, int, str]]) -> None:
        
        super(DeepRegressionModel, self).__init__()
        
    
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    

    @staticmethod
    @abc.abstractmethod
    def initialize_weights(m: nn.Module) -> None:
        pass
    
    
    @staticmethod
    @abc.abstractmethod
    def takes_time_series() -> bool:
        pass


    def __str__(self) -> str:
        
        if isinstance(self.input_size, int):
            return f"[*] {self.__class__.__name__} summary:\n" \
                f"{torchinfo.summary(self, [1, self.input_size], verbose=0, device=self.device)}\n"
            
        else:
            return f"[*] {self.__class__.__name__} summary:\n" \
                f"{torchinfo.summary(self, [1, *self.input_size], verbose=0, device=self.device)}\n"


class EnsembleDeepRegressionModel(abc.ABC):
    
    def __init__(self, args: typing.Dict[str, typing.Union[float, int, str]]) -> None:
        
        super(EnsembleDeepRegressionModel, self).__init__()
        
        self.id = uuid.uuid4()
        
        self.device         = args["device"]
        self.ensemble_count = args["ensemble_count"]
        self.model_class    = m_constants.MODELS[args["model"]]
        self.models         = nn.ModuleList([self.model_class(args) for _ in range(self.ensemble_count)])
  
    
    def __str__(self) -> str:
        return f"[*] Ensemble of {self.ensemble_count} {self.model_class.__name__} models...\n" \
               f"{self.models[0]}\n"
                   
    
    def run_step(self, data_loader: torch.utils.data.DataLoader, mode: str) -> float:
        
        assert mode in ['train', 'val', 'test'], "[!] Invalid 'run_step' mode. Must be 'train', 'val', or 'test'."

        if mode == 'train':
            for model in self.models:
                model.train()
        else:
            for model in self.models:
                model.eval()

        all_outputs = []
        all_targets = []
        train_losses = []

        with torch.set_grad_enabled(mode == 'train'):
            
            for batch_id, (samples, targets) in enumerate(data_loader):
                
                batch_outputs = []
                
                for model in self.models:
                    
                    # Moved per model to avoid issues in computational graphs
                    model_samples = samples.to(model.device, non_blocking=True)
                    model_targets = targets.to(model.device, non_blocking=True)

                    if mode == 'train':
                        model.optimizer.zero_grad(set_to_none=True)

                    outputs = model(model_samples)

                    if mode == 'train':
                        loss = model.criterion(outputs, model_targets)
                        loss.backward()
                        model.optimizer.step()
                        train_losses.append(loss.item())

                    else:
                        batch_outputs.append(outputs)

                if mode != 'train':
                    all_outputs.extend(self.average_outputs(batch_outputs).tolist())
                    all_targets.extend(targets.tolist())

                del(samples, targets, outputs, batch_outputs)

        if mode == 'train':
            return np.mean(train_losses)
        
        elif mode == 'val':
            return nn.functional.l1_loss(torch.tensor(all_outputs), torch.tensor(all_targets)).item()
        
        else: # test
            return m_metrics.compute_metrics(all_targets, all_outputs)


    def average_outputs(self, batch_outputs: typing.List[torch.Tensor]) -> torch.Tensor:
        
        stacked_outputs = torch.stack(batch_outputs, dim=0)
        mean_outputs    = torch.mean(stacked_outputs, dim=0)
        
        return mean_outputs
    
    
    def lr_scheduler_step(self, val_loss: float, warmup: bool) -> None:
        
        for model in self.models:
            
            if warmup and model.lr_scheduler_warmup is not None:
                # Always linear
                model.lr_scheduler_warmup.step()
                
            elif not warmup and model.lr_scheduler is not None:
                if isinstance(model.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    model.lr_scheduler.step(val_loss)
                elif isinstance(model.lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    model.lr_scheduler.step()
                elif isinstance(model.lr_scheduler, torch.optim.lr_scheduler.LambdaLR):
                    model.lr_scheduler.step()
                else:
                    raise NotImplementedError(f"[!] Learning rate scheduler {model.lr_scheduler} not implemented.")


    def load_model(self, model_path: str) -> None:

        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(f"{model_path}_{i}.pth", map_location=self.device))


    def save_model(self, model_path: str) -> None:
        
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), f"{model_path}_{i}.pth")
