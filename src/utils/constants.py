__author__ = "alvrogd"


import typing

import models.bilstm as m_bilstm
import models.fcn as m_fcn
import models.inceptiontime as m_inceptiontime
import models.mlp as m_mlp
import models.residualnet as m_residualnet
import models.resnet18t as m_resnet18t
import models.rocket as m_rocket
import models.templates as m_templates
import models.transformer as m_transformer


## Datasets

BANDS: typing.List[str] = ["blue", "green", "red", "nir", "nirb", "re1", "re2", "re3", "swir1", "swir2"]

TIME_STEPS: int = 16

PREDICTORS: typing.List[str] = [f"{band}_{i}" for i in range(TIME_STEPS) for band in BANDS]

STUDY_VARS: typing.List[str] = ["Shannon", "Simpson", "SpecRichness"]


## Models

MODELS: typing.Dict[str, typing.Type[m_templates.DeepRegressionModel]] = {
    "BiLSTM":        m_bilstm.BiLSTM,
    "FCN":           m_fcn.FCN,
    "InceptionTime": m_inceptiontime.InceptionTime,
    "MLP":           m_mlp.MLP,
    "ResidualNet":   m_residualnet.ResidualNet,
    "ResNet18T":     m_resnet18t.ResNet18T,
    "Rocket":        m_rocket.Rocket,
    "Transformer":   m_transformer.Transformer,
}
