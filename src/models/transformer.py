__author__ = "alvrogd"


import typing

import numpy as np
import torch
import torch.nn as nn
import torchinfo

import models.templates as m_templates
import utils.constants as m_constants


def get_sinusoid_encoding_table(num_positions, num_dimensions, padding_dimension=0):

    # Some helpers

    def calculate_angle(position, dimension):
        return position / np.power(10000, 2 * (dimension // 2) / num_dimensions)

    def get_position_angle_vector(position):
        return [calculate_angle(position, dimension) for dimension in range(num_dimensions)]


    # Create the encoding table

    sinusoid_table = np.array([get_position_angle_vector(position) for position in range(num_positions)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # Even dimensions
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # Odd dimensions

    # The padding dimension gets zeroed
    sinusoid_table[padding_dimension] = 0.

    if torch.cuda.is_available():
        return torch.FloatTensor(sinusoid_table).cuda()
    else:
        return torch.FloatTensor(sinusoid_table)
    
    # Output shape: (n_position, d_hid)


class Transformer(m_templates.DeepRegressionModel):
    # Based on: https://github.com/MarcCoru/crop-type-mapping/blob/fcc040181de93ee466e525150983ddd6822f86d5/src/models/TransformerEncoder.py
    #
    # Default parameters:
    #   batch_size=32, epochs=1500, warmup_epochs=1

    def __init__(self, args: typing.Dict[str, typing.Union[float, int, str]]) -> None:

        super(Transformer, self).__init__(args)
        
        # This network can interpret predictors as a time series
        self.input_size        = (len(m_constants.BANDS), m_constants.TIME_STEPS)
        self.input_bands       = self.input_size[0]
        self.input_time_steps  = self.input_size[1]
        self.device            = args["device"]


        self.input_norm      = nn.LayerNorm(self.input_bands)
        self.input_conv      = nn.Conv1d(self.input_bands, 64, kernel_size=1)
        self.input_conv_norm = nn.LayerNorm(64)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=256,
            dropout=0.015
        )
        self.encoder_norm = nn.LayerNorm(64)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=5,
            norm=self.encoder_norm,
            
        )

        self.head = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1, bias=False),
        )
        
        self.position_encoder = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self.input_time_steps + 1, 64),
            freeze=True
        )


        self.apply(self.initialize_weights)

        # The model must be moved to the GPU before constructing any optimizers
        self.to(self.device)

        self.criterion           = nn.L1Loss()
        self.optimizer           = torch.optim.Adam(self.parameters(), lr=2e-3, betas=(0.9, 0.999), eps=1e-9, weight_decay=2e-4)
        def lr_scheduler_lambda(epoch):
            epoch += 1 # To avoid 0-epoch issues
            return min(
                epoch ** (-0.5),
                epoch * (100 ** (-1.5))
            )
        self.lr_scheduler        = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_scheduler_lambda)
        self.lr_scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: min(1.0, epoch / args["warmup_epochs"]))


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Input shape: (batch_size, bands, length)
        
        # (batch_size, length, bands)
        x = x.transpose(1, 2)
        x = self.input_norm(x)
        
        # (batch_size, bands, length)
        x = x.transpose(1, 2)
        x = self.input_conv(x)
        
        # (batch_size, length, bands)
        x = x.transpose(1, 2)
        x = self.input_conv_norm(x)
        
        # Add positional encoding
        batch_size, length, _ = x.shape
        positions             = torch.arange(1, length + 1, dtype=torch.int64, device=self.device).expand(batch_size, length)
        x                     = x + self.position_encoder(positions)

        # (length, batch_size, bands)
        x = x.transpose(0, 1)
        x = self.encoder(x)

        # Back to (batch_size, bands, length)
        x = x.permute(1, 2, 0)
        features = self.head(x)

        return features


    @staticmethod
    def initialize_weights(m: nn.Module) -> None:

        # All modules use their default weight initialization
        pass


    @staticmethod
    def takes_time_series() -> bool:

        return True
