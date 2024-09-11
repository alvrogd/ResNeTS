__author__ = "alvrogd"


import typing
import uuid

import sktime.regression.kernel_based
import torch

import utils.metrics as m_metrics


class Rocket():
    """
    This class implements just the essential procedures to run a single instance of the sktime.RocketRegressor,
    enabling its evaluation without modifying the already present training code in main.py.
    
    It's not an escalable solution, but I currently don't plan on testing more non-deep learning architectures.
    """
        
    def __init__(self, args: typing.Dict[str, typing.Union[float, int, str]]) -> None:
        
        super(Rocket, self).__init__()
        
        self.id = uuid.uuid4()
        
        self.num_kernels = args["num_kernels"]
        
        self.model = sktime.regression.kernel_based.RocketRegressor(num_kernels=self.num_kernels,
                                                                    rocket_transform="rocket", use_multivariate="yes",
                                                                    n_jobs=-1, random_state=args["seed"])
  
    
    def __str__(self) -> str:
        return f"[*] Rocket regressor...\n" \
               f"Num kernels: {self.num_kernels}\n" \
               f"Total params: -1\n"
                   
    
    def run_step(self, data_loader: torch.utils.data.DataLoader, mode: str) -> float:
        
        assert mode in ['train', 'val', 'test'], "[!] Invalid 'run_step' mode. Must be 'train', 'val', or 'test'."

        all_samples = []
        all_targets = []
        
        # RocketRegressor expects to receive all samples at the same time
        for samples, targets in data_loader:
            all_samples.append(samples)
            all_targets.append(targets)
            
        all_samples = torch.cat(all_samples, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # And in np.array format:
        #   - samples: [n_instances, n_dimensions, series_length]
        #   - targets: [n_instances]
        all_samples = all_samples.numpy()
        all_targets = all_targets.squeeze().numpy()
        assert len(all_targets.shape) == 1, "[!] Invalid 'all_targets' shape. It cannot contain multiple dimensions."
        
        if mode == 'train':
            self.model.fit(all_samples, all_targets)
            return 0
        
        elif mode == 'val':
            print("\n[!] RocketModel ignores 'val' run_step.")
            return 0
        
        else:  # test
            all_outputs = self.model.predict(all_samples)
            assert len(all_outputs.shape) == 1, "[!] Invalid 'all_outputs' shape. It cannot contain multiple dimensions."
            
            all_outputs = all_outputs.tolist()
            all_targets = all_targets.tolist()
            
            return m_metrics.compute_metrics(all_targets, all_outputs)

    
    def lr_scheduler_step(self, val_loss: float, warmup: bool) -> None:
        pass


    @staticmethod
    def takes_time_series() -> bool:
        return True


    def load_model(self, model_path: str) -> None:

        self.model = sktime.regression.kernel_based.RocketRegressor.load_from_path(f"{model_path}.zip")


    def save_model(self, model_path: str) -> None:
        
        self.model.save(f"{model_path}")
