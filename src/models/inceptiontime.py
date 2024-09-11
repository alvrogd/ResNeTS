__author__ = "alvrogd"


import typing

import torch
import torch.nn as nn
import torchinfo

import models.templates as m_templates
import utils.constants as m_constants


class InceptionTime(m_templates.DeepRegressionModel):
    # Based on: https://github.com/hfawaz/InceptionTime/tree/470ce144c1ba43b421e72e1d216105db272e513f/classifiers
    #
    # Default parameters:
    #   batch_size=32, epochs=1500, warmup_epochs=1, bottleneck_factor=4, num_filters=64, original_training=True
    #
    # Improved training parameters:
    #   batch_size=32, beta1=0.9, beta2=0.999, epochs=1500, eps=0.01, lr=0.001, warmup_epochs=150, weight_decay=0.001, bottleneck_factor=4, num_filters=64, original_training=False

    def __init__(self, args: typing.Dict[str, typing.Union[float, int, str]]) -> None:

        super(InceptionTime, self).__init__(args)

        # This network can interpret predictors as a time series
        self.input_size        = (len(m_constants.BANDS), m_constants.TIME_STEPS)
        self.input_bands       = self.input_size[0]
        self.input_time_steps  = self.input_size[1]
        self.device            = args["device"]
        
        self.bottleneck_factor = args["bottleneck_factor"]
        self.num_filters       = args["num_filters"]
        
        self.beta1                = args["beta1"]
        self.beta2                = args["beta2"]
        self.eps                  = args["eps"]
        self.lr                   = args["lr"]
        self.original_training    = args["original_training"]
        self.weight_decay         = args["weight_decay"]

       
        # a) 2 residual blocks
        # b) 3 inception modules per block
        # c) A skip-connection between blocks
        # d) Global Average Pooling + Fully Connected at the end
        self.mod1   = InceptionTimeModule((self.input_bands,     self.input_time_steps), self.bottleneck_factor, self.num_filters)
        self.mod2   = InceptionTimeModule((self.num_filters * 4, self.input_time_steps), self.bottleneck_factor, self.num_filters)
        self.mod3   = InceptionTimeModule((self.num_filters * 4, self.input_time_steps), self.bottleneck_factor, self.num_filters)
        self.short1 = InceptionTimeShortcut((self.input_bands,   self.input_time_steps), (self.num_filters * 4, self.input_time_steps))
                
        self.mod4   = InceptionTimeModule((self.num_filters * 4,   self.input_time_steps), self.bottleneck_factor, self.num_filters)
        self.mod5   = InceptionTimeModule((self.num_filters * 4,   self.input_time_steps), self.bottleneck_factor, self.num_filters)
        self.mod6   = InceptionTimeModule((self.num_filters * 4,   self.input_time_steps), self.bottleneck_factor, self.num_filters)
        self.short2 = InceptionTimeShortcut((self.num_filters * 4, self.input_time_steps), (self.num_filters * 4, self.input_time_steps))
            
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(self.num_filters * 4, 1)
               
        
        self.apply(self.initialize_weights)

        # The model must be moved to the GPU before constructing any optimizers
        self.to(self.device)
        
        self.criterion = nn.L1Loss()
        if self.original_training:
            self.optimizer    = torch.optim.Adam(self.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-07, weight_decay=0.0)
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=50, min_lr=0.00005)
        else:
            self.optimizer    = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), eps=self.eps, weight_decay=self.weight_decay)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args["epochs"], eta_min=0.0)
        self.lr_scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: min(1.0, epoch / args["warmup_epochs"]))


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output = self.mod1(x)
        output = self.mod2(output)
        output = self.mod3(output)
        output = self.short1(output, x)
        
        x      = output
        output = self.mod4(output)
        output = self.mod5(output)
        output = self.mod6(output)
        output = self.short2(output, x)
        
        output = self.gap(output)
        output = output.view(output.shape[0], -1)
        output = self.fc(output)

        return output


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


class InceptionTimeModule(nn.Module):
    
    def __init__(self, input_size: typing.Tuple[int, int], bottleneck_factor: int, num_filters: int) -> None:
        
        super(InceptionTimeModule, self).__init__()
        
        self.input_size       = input_size
        self.input_bands      = self.input_size[0]
        self.input_time_steps = self.input_size[1]
        self.bottleneck_bands = self.input_bands // bottleneck_factor
        self.num_filters      = num_filters
        
        
        # 1st path: bottleneck + 3 convolutions
        
        if self.input_bands > 1:
            self.path1_b = nn.Conv1d(self.input_bands, self.bottleneck_bands, kernel_size=1, stride=1, padding="same", bias=False)
        else:
            self.path1_b = nn.Identity()
        
        self.path1_c1 = nn.Conv1d(self.bottleneck_bands, self.num_filters, kernel_size=8, stride=1, padding="same", bias=False)
        self.path1_c2 = nn.Conv1d(self.bottleneck_bands, self.num_filters, kernel_size=4, stride=1, padding="same", bias=False)
        self.path1_c3 = nn.Conv1d(self.bottleneck_bands, self.num_filters, kernel_size=2, stride=1, padding="same", bias=False)
            
        # 2nd path: max-pooling + bottleneck
        self.path2_p = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.path2_b = nn.Conv1d(self.input_bands, self.num_filters, kernel_size=1, stride=1, padding="same", bias=False)
        
        # Merge paths
        self.merge = nn.Sequential(        
            # Concatenation in forward()
            nn.BatchNorm1d(self.num_filters * 4, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
            
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        path1 = self.path1_b(x)
        path1 = [self.path1_c1(path1), self.path1_c2(path1), self.path1_c3(path1)]
        
        path2 = self.path2_p(x)
        path2 = [self.path2_b(path2)]
        
        # Concatenate along the channel dimension
        merge = torch.cat(path1 + path2, dim=-2)
        merge = self.merge(merge)
        
        return merge
        
        
class InceptionTimeShortcut(nn.Module):
    
    def __init__(self, input_size: typing.Tuple[int, int], output_size: typing.Tuple[int, int]) -> None:
        
        super(InceptionTimeShortcut, self).__init__()
        
        self.input_size        = input_size
        self.input_bands       = self.input_size[0]
        self.input_time_steps  = self.input_size[1]
        self.output_size       = output_size
        self.output_bands      = self.output_size[0]
        self.output_time_steps = self.output_size[1]
        

        self.mapping = nn.Sequential(
            nn.Conv1d(self.input_bands, self.output_bands, kernel_size=1, stride=1, padding="same", bias=False),
            nn.BatchNorm1d(self.output_bands, eps=1e-3, momentum=0.01)
        )
        
        # Addition in forward()
        
        self.activation = nn.Sequential(
            nn.ReLU()
        )
        
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        
        mapping = self.mapping(y)
        output  = x + mapping
        output  = self.activation(output)
                          
        return output
