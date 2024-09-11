__author__ = "alvrogd"


import typing

import torch
import torch.nn as nn
import torchinfo

import models.templates as m_templates
import utils.constants as m_constants


class ResNet18T(m_templates.DeepRegressionModel):
    # Based on: https://github.com/rwightman/pytorch-image-models/blob/624266148d8fa5ddb22a6f5e523a53aaf0e8a9eb/timm/models/resnet.py
    #
    # Default parameters:
    #   batch_size=32, beta1=0.9, beta2=0.999, epochs=1500, eps=1e-6, lr=0.001, warmup_epochs=150, weight_decay=0.001, num_blocks_per_stage=[1, 1, 1, 1], num_channels=[64, 128, 256, 512], kernel_size=5, original_training=False, shortcut_pooling=True, stem_channels=96, strides=[1, 1, 2, 1]

    def __init__(self, args: typing.Dict[str, typing.Union[float, int, str]]) -> None:

        super(ResNet18T, self).__init__(args)

        # This network can interpret predictors as a time series
        self.input_size        = (len(m_constants.BANDS), m_constants.TIME_STEPS)
        self.input_bands       = self.input_size[0]
        self.input_time_steps  = self.input_size[1]
        self.device            = args["device"]
        
        self.beta1                = args["beta1"]
        self.beta2                = args["beta2"]
        self.eps                  = args["eps"]
        self.lr                   = args["lr"]
        self.num_blocks_per_stage = args["num_blocks_per_stage"]
        self.num_channels         = args["num_channels"]
        self.kernel_size          = args["kernel_size"]
        self.original_training    = args["original_training"]
        self.shortcut_pooling     = args["shortcut_pooling"]
        self.stem_channels        = args["stem_channels"]
        self.strides              = args["strides"]
        self.weight_decay         = args["weight_decay"]
        
        
        # The stem convolution will just map the input to a higher number of bands
        # No spatial downsampling as our time series is already quite short
        self.stem = nn.Sequential(
            nn.Conv1d(self.input_bands, self.stem_channels, kernel_size=1, stride=1, padding="same", bias=False),
            nn.BatchNorm1d(self.stem_channels),
            nn.ReLU()
        )

        self.stage1 = ResNetStage(input_bands=self.stem_channels,   output_bands=self.num_channels[0], number_blocks=self.num_blocks_per_stage[0], stride=self.strides[0], kernel_size=self.kernel_size, shortcut_pooling=self.shortcut_pooling)
        self.stage2 = ResNetStage(input_bands=self.num_channels[0], output_bands=self.num_channels[1], number_blocks=self.num_blocks_per_stage[1], stride=self.strides[1], kernel_size=self.kernel_size, shortcut_pooling=self.shortcut_pooling)
        self.stage3 = ResNetStage(input_bands=self.num_channels[1], output_bands=self.num_channels[2], number_blocks=self.num_blocks_per_stage[2], stride=self.strides[2], kernel_size=self.kernel_size, shortcut_pooling=self.shortcut_pooling)
        self.stage4 = ResNetStage(input_bands=self.num_channels[2], output_bands=self.num_channels[3], number_blocks=self.num_blocks_per_stage[3], stride=self.strides[3], kernel_size=self.kernel_size, shortcut_pooling=self.shortcut_pooling)
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.num_channels[3], 1)
        )


        self.apply(self.initialize_weights)

        # The model must be moved to the GPU before constructing any optimizers
        self.to(self.device)
        
        self.criterion = nn.L1Loss()
        if self.original_training:
            self.optimizer    = torch.optim.SGD(self.parameters(), lr=0.0125, momentum=0.9, weight_decay=1e-4)
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1, patience=50)
        else:
            self.optimizer    = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), eps=self.eps, weight_decay=self.weight_decay)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args["epochs"], eta_min=0.0)
        self.lr_scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: min(1.0, epoch / args["warmup_epochs"]))
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out = self.stem(x)
        
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        
        out = self.head(out)
        
        return out
        

    @staticmethod
    def initialize_weights(m: nn.Module) -> None:

        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


    @staticmethod
    def takes_time_series() -> bool:

        return True


class ResNetStage(nn.Module):
    
    def __init__(self, input_bands: int, output_bands: int, number_blocks: int, stride: int, kernel_size: int, shortcut_pooling = True) -> None:
        
        super(ResNetStage, self).__init__()
        
        self.input_bands      = input_bands
        self.output_bands     = output_bands
        self.number_blocks    = number_blocks
        self.stride           = stride
        self.kernel_size      = kernel_size
        self.shortcut_pooling = shortcut_pooling
        
        
        layers = []
        
        if self.input_bands != self.output_bands or self.stride != 1:
            
            if self.shortcut_pooling:
                mapping = nn.Sequential(
                    nn.AvgPool1d(kernel_size=self.stride, stride=self.stride, padding=0),
                    nn.Conv1d(self.input_bands, self.output_bands, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm1d(self.output_bands)
                )
                
            else:
                mapping = nn.Sequential(
                    nn.Conv1d(self.input_bands, self.output_bands, kernel_size=1, stride=self.stride, padding=0, bias=False),
                    nn.BatchNorm1d(self.output_bands)
                )
                
        else:
            mapping = nn.Identity()
        
        layers.append(
            ResNetBlockBasic(self.input_bands, self.output_bands, stride=self.stride, mapping=mapping, kernel_size=self.kernel_size)
        )
        
        for _ in range(1, self.number_blocks):
            layers.append(
                ResNetBlockBasic(self.output_bands, self.output_bands, stride=1, mapping=nn.Identity(), kernel_size=self.kernel_size)
            )
        
        self.layers = nn.Sequential(*layers)
            
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.layers(x)
        
        
class ResNetBlockBasic(nn.Module):
    
    def __init__(self, input_bands: int, output_bands: int, stride: int, mapping: nn.Module, kernel_size: int) -> None:
        
        super(ResNetBlockBasic, self).__init__()
        
        self.input_bands      = input_bands
        self.output_bands     = output_bands
        self.stride           = stride
        self.mapping          = mapping
        self.kernel_size      = kernel_size
        
               
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.input_bands, self.output_bands, kernel_size=self.kernel_size, stride=self.stride, padding=self.kernel_size // 2, bias=False),
            nn.BatchNorm1d(self.output_bands),
            nn.ReLU(),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.output_bands, self.output_bands, kernel_size=self.kernel_size, stride=1, padding="same", bias=False),
            nn.BatchNorm1d(self.output_bands),
        )
        
        self.conv2_activation = nn.ReLU()          
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        out = out + self.mapping(identity)
        out = self.conv2_activation(out)
        
        return out
