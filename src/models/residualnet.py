__author__ = "alvrogd"


import typing

import torch
import torch.nn as nn
import torchinfo

import models.templates as m_templates
import utils.constants as m_constants


class ResidualNet(m_templates.DeepRegressionModel):
    # Based on: https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline/blob/3ecc8971fa768bd01ada7f3a613688255e6256c2/ResNet.py
    #
    # Default parameters:
    #   batch_size=32, epochs=1500, warmup_epochs=1

    def __init__(self, args: typing.Dict[str, typing.Union[float, int, str]]) -> None:

        super(ResidualNet, self).__init__(args)

        # This network can interpret predictors as a time series
        self.input_size       = (len(m_constants.BANDS), m_constants.TIME_STEPS)
        self.input_bands      = self.input_size[0]
        self.input_time_steps = self.input_size[1]
        self.device           = args["device"]


        self.block1 = ResidualNetBlock(input_bands=self.input_bands, output_bands=64)
        
        self.block2 = ResidualNetBlock(input_bands=64, output_bands=128)
        
        self.block3 = ResidualNetBlock(input_bands=128, output_bands=128)
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
        )

        
        self.apply(self.initialize_weights)

        # The model must be moved to the GPU before constructing any optimizers
        self.to(self.device)
        
        self.criterion           = nn.L1Loss()
        self.optimizer           = torch.optim.Adam(self.parameters(), lr=2e-3, betas=(0.9, 0.999), eps=1e-7, weight_decay=0.0, amsgrad=False)
        self.lr_scheduler        = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=50, min_lr=2e-4, verbose=True)
        self.lr_scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: min(1.0, epoch / args["warmup_epochs"]))


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.head(out)

        return out


    @staticmethod
    def initialize_weights(m: nn.Module) -> None:

        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

         
    @staticmethod
    def takes_time_series() -> bool:

        return True


class ResidualNetBlock(nn.Module):
    
    def __init__(self, input_bands: int, output_bands: int) -> None:
        
        super(ResidualNetBlock, self).__init__()
        
        self.input_bands  = input_bands
        self.output_bands = output_bands


        if input_bands != output_bands:
            self.mapping = nn.Sequential(
                nn.Conv1d(self.input_bands, self.output_bands, kernel_size=1, stride=1, padding="same", bias=False),
                nn.BatchNorm1d(self.output_bands, eps=1e-3, momentum=0.01)
            )
        else:
            self.mapping = nn.BatchNorm1d(self.output_bands, eps=1e-3, momentum=0.01)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.input_bands, out_channels=self.output_bands, kernel_size=8, stride=1, padding="same", bias=False),
            nn.BatchNorm1d(self.output_bands, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.output_bands, out_channels=self.output_bands, kernel_size=5, stride=1, padding="same", bias=False),
            nn.BatchNorm1d(self.output_bands, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.output_bands, out_channels=self.output_bands, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm1d(self.output_bands, eps=1e-3, momentum=0.01),
        )

        self.conv3_activation = nn.ReLU()            
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        out = out + self.mapping(identity)
        out = self.conv3_activation(out)
        
        return out
