__author__ = "alvrogd"


import typing

import torch
import torch.nn as nn
import torchinfo

import models.templates as m_templates
import utils.constants as m_constants


class BiLSTM(m_templates.DeepRegressionModel):
    # Based on: https://github.com/MarcCoru/crop-type-mapping/blob/fcc040181de93ee466e525150983ddd6822f86d5/src/models/rnn.py
    #
    # Default parameters:
    #   batch_size=32, epochs=1500, warmup_epochs=1

    def __init__(self, args: typing.Dict[str, typing.Union[float, int, str]]) -> None:

        super(BiLSTM, self).__init__(args)

        # This network can interpret predictors as a time series
        self.input_size        = (len(m_constants.BANDS), m_constants.TIME_STEPS)
        self.input_bands       = self.input_size[0]
        self.input_time_steps  = self.input_size[1]
        self.device            = args["device"]


        self.input_norm = nn.LayerNorm(self.input_bands)
        
        self.lstm = nn.LSTM(
            input_size=self.input_bands,
            hidden_size=128,
            num_layers=3,
            bias = False,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        
        # 2 (bidirectional) * 3 (num_layers) * 128 (hidden_size)
        self.num_features = 2 * 3 * 128
        self.head = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Linear(self.num_features, 1)
        )
        
        
        self.apply(self.initialize_weights)

        # The model must be moved to the GPU before constructing any optimizers
        self.to(self.device)
        
        self.criterion           = nn.L1Loss()
        self.optimizer           = torch.optim.Adam(self.parameters(), lr=5e-3 * 32/256, betas=(0.9, 0.999), eps=1e-8, weight_decay=2e-5)
        self.lr_scheduler        = None
        self.lr_scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: min(1.0, epoch / args["warmup_epochs"]))


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # nn.LSTM expects (batch_size, length, bands) order
        x = x.transpose(1, 2)

        input = self.input_norm(x)
        
        # The regression features will be extracted from the final cell state
        _, (_, c_n) = self.lstm(input)
        
        # "batch_first" is ignored in the cell state
        c_n = c_n.transpose(0, 1)
        
        batch_size = c_n.shape[0]
        c_n = c_n.reshape(batch_size, -1)
        
        features = self.head(c_n)

        return features


    @staticmethod
    def initialize_weights(m: nn.Module) -> None:

        # All modules use their default weight initialization
        pass


    @staticmethod
    def takes_time_series() -> bool:

        return True
