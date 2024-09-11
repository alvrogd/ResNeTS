__author__ = "alvrogd"


import typing

import torch
import torch.nn as nn
import torchinfo

import models.templates as m_templates
import utils.constants as m_constants


class MLP(m_templates.DeepRegressionModel):
    # Based on: https://github.com/Havi-muro/SeBAS_project/blob/d89a4909efde7862b168cd8c8bc2dad130655f86/gkfold_DNN.py
    #
    # Default parameters:
    #   batch_size=32, epochs=1500, warmup_epochs=1

    def __init__(self, args: typing.Dict[str, typing.Union[float, int, str]]) -> None:

        super(MLP, self).__init__(args)

        # This network cannot analyze a time series per se, so it takes all data points at once
        self.input_size = len(m_constants.PREDICTORS)
        self.device     = args["device"]
        

        # L1 regularization ommited in layers 1, 2 and 3

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU()
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.fc4 = nn.Sequential(
            nn.Linear(64, 1)
        )
        

        self.apply(self.initialize_weights)

        # The model must be moved to the GPU before constructing any optimizers
        self.to(self.device)
        
        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-07, weight_decay=0)
        self.lr_scheduler = None
        self.lr_scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: min(1.0, epoch / args["warmup_epochs"]))


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)

        return out


    @staticmethod
    def initialize_weights(m: nn.Module) -> None:

        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


    @staticmethod
    def takes_time_series() -> bool:

        return False
