# ResNeTS: a ResNet for Time Series Analysis of Sentinel-2 Data Applied to Grassland Plant-Biodiversity Prediction

Official PyTorch implementation of ResNeTS, introduced in:

> Á. G. Dieste et al., "ResNeTS: a ResNet for Time Series Analysis of
> Sentinel-2 Data Applied to Grassland Plant-Biodiversity Prediction," in IEEE
> Journal of Selected Topics in Applied Earth Observations and Remote Sensing,
> doi:
> [10.1109/JSTARS.2024.3454271](https://doi.org/10.1109/JSTARS.2024.3454271).

For questions or inquiries, please contact:
[alvaro.goldar.dieste@usc.es](mailto:alvaro.goldar.dieste@usc.es)

## Table of contents

- [ResNeTS: a ResNet for Time Series Analysis of Sentinel-2 Data Applied to Grassland Plant-Biodiversity Prediction](#resnets-a-resnet-for-time-series-analysis-of-sentinel-2-data-applied-to-grassland-plant-biodiversity-prediction)
  - [Table of contents](#table-of-contents)
  - [About ResNeTS](#about-resnets)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Downloading the code](#downloading-the-code)
    - [Downloading the dataset](#downloading-the-dataset)
    - [Running the project](#running-the-project)
  - [Usage](#usage)
  - [License](#license)
  - [Citation](#citation)

## About ResNeTS

ResNeTS is an adaptation of the ResNet computer vision architecture for time
series analysis of Sentinel-2 data. By favoring a streamlined and efficient
design, ResNeTS improves accuracy over state-of-the-art architectures like
InceptionTime and Transformers, while also reducing computational costs.

For further details, please refer to the associated research paper. A brief
overview of the work can be found below:

> Analyzing time series from remote sensing data can aid in understanding
> spectral-temporal phenomena in ecosystems, such as the seasonal variation of
> plant components. Lately, deep learning has emerged as a strong method for
> mapping environmental variables from this data due to its exceptional
> predictive capabilities. This work studies the adaptation of the ResNet
> computer vision architecture for time series analysis of Sentinel-2 data. The
> resulting deep learning architecture, ResNeTS, stacks sequential convolutions
> to build a deep and narrow network, aligning with the design principles of
> leading convolutional architectures in computer vision. Experiments were
> carried out for predicting different plant-biodiversity indices, namely
> species richness, and Shannon and Simpson indices, for temperate grassland
> ecosystems. The results show that ResNeTS can achieve moderate improvements
> in terms of accuracy compared to other state-of-the-art architectures, such
> as InceptionTime (up to +0.021 r2), with reduced computational costs owing to
> its streamlined architecture.

## Installation

Follow the instructions below to set up this project for experimentation.

### Prerequisites

This project leverages Docker to ensure a reproducible environment. To get
started, ensure your machine supports the following tools:

- [Docker](https://docs.docker.com/engine/install/).

- [NVIDIA Container
  Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (optional but highly recommended for CUDA acceleration).

### Downloading the code

Clone the repository to your local machine:

```bash
git clone https://github.com/alvrogd/ResNeTS.git
```

The code will be provided to the Docker container through volume mapping.

### Downloading the dataset

The dataset used in this work is already located in the `data/` directory. It
contains Sentinel-2 time series data along with the corresponding biodiversity
indices in a `.xlsx` format.

### Running the project

To build the Docker image, run the following script:

```bash
cd dockerfiles/ && bash ./build_container.sh
```

Note that the image is large (~18.6 GB), so downloading dependencies may take
some time.

Before launching the container, update the volume mapping in
`dockerfiles/launch_container.sh` to fit your particular set-up:

```diff
docker run \
     --shm-size=1g \
     --rm \
     -p 6006:6006 \
-    -v /home/alvaro.goldar/ResNeTS:/opt/ResNeTS \
+    -v /full/path/to/ResNeTS:/opt/ResNeTS \
     resnets
```

Feel free to change any other arguments as needed.

Now, launch the container and attach a `bash` terminal:

```bash
./launch_container.sh bash
```

## Usage

The container includes the Guild AI tool, which helps track experiments,
compare results, and automate runs. To learn more about the many features of
Guild AI, such as batch files and others, please refer to its
[documentation](https://my.guild.ai/docs).

Guild AI parses available arguments and uses their default values unless
otherwise specified:

```txt
root@85d5877463d3:/opt/ResNeTS# guild run
You are about to run main
  batch_size: 32
  beta1: 0.9
  beta2: 0.999
  bottleneck_factor: 4
  ensemble_count: 1
  epochs: 1500
  eps: 1.0e-06
  kernel_size: 5
  lr: 0.001
  model: ResNet18T
  num_blocks_per_stage: 1 1 1 1
  num_channels: 64 64 64 64
  num_filters: 64
  num_kernels: 15000
  original_training: no
  seed: 42
  shortcut_pooling: yes
  split_procedure: split_by_plot
  stem_channels: 96
  strides: 1 1 2 1
  study_var: SpecRichness
  warmup_epochs: 150
  weight_decay: 0.001
Continue? (Y/n)
```

You can change any parameter by appending its new value to the command. For
instance, to predict the Shannon index instead of Species Richness, use:

```txt
root@85d5877463d3:/opt/ResNeTS# guild run study_var=Shannon
You are about to run main
  <...>
  study_var: Shannon
  <...>
Continue? (Y/n)
```

As an example, let's train the ResNeTS model for predicting Species Richness:

```txt
root@85d5877463d3:/opt/ResNeTS# guild run model=ResNet18T study_var=SpecRichness
You are about to run main
  batch_size: 32
  beta1: 0.9
  beta2: 0.999
  bottleneck_factor: 4
  ensemble_count: 1
  epochs: 1500
  eps: 1.0e-06
  kernel_size: 5
  lr: 0.001
  model: ResNet18T
  num_blocks_per_stage: 1 1 1 1
  num_channels: 64 64 64 64
  num_filters: 64
  num_kernels: 15000
  original_training: no
  seed: 42
  shortcut_pooling: yes
  split_procedure: split_by_plot
  stem_channels: 96
  strides: 1 1 2 1
  study_var: SpecRichness
  warmup_epochs: 150
  weight_decay: 0.001
Continue? (Y/n) 
Resolving file:data/
[*] Arguments: {'device': 'cuda:0', 'seed': 42, 'batch_size': 32, 'split_procedure': 'split_by_plot', 'study_var': 'SpecRichness', 'beta1': 0.9, 'beta2': 0.999, 'ensemble_count': 1, 'epochs': 1500, 'eps': 1e-06, 'lr': 0.001, 'model': 'ResNet18T', 'warmup_epochs': 150, 'weight_decay': 0.001, 'bottleneck_factor': 4, 'num_filters': 64, 'num_blocks_per_stage': [1, 1, 1, 1], 'num_channels': [64, 64, 64, 64], 'kernel_size': 5, 'shortcut_pooling': True, 'stem_channels': 96, 'strides': [1, 1, 2, 1], 'original_training': False, 'num_kernels': 15000}
[*] Fold 1/5
[*] Ensemble of 1 ResNet18T models...
<...>
[*] Warmup epoch: 1/150 - Train loss: 30.4713                       
[*] Warmup epoch: 10/150 - Train loss: 29.9158
<...>
[*] Epoch: 10/1500 - Train loss: 2.4998 - Val loss: 5.2823                 
[*] Epoch: 20/1500 - Train loss: 1.8214 - Val loss: 4.9980
<...>
[*] Epoch: 370/1500 - Train loss: 0.6792 - Val loss: 4.7428                  
[*] Early stopping at epoch 380                                              
[*] Training the model...:  25%|██▌       | 379/1500 [01:04<03:10,  5.89it/s]
[*] Training time: 88.24 s
[*] Testing the best model...
[*] Testing time: 0.04 s
[*] Fold 2/5
<...>
[*] Fold 3/5
<...>
[*] Fold 4/5
<...>
[*] Fold 5/5
<...>
[*] Epoch: 350/1500 - Train loss: 1.0331 - Val loss: 4.2093                  
[*] Early stopping at epoch 360                                              
[*] Training the model...:  24%|██▍       | 359/1500 [01:01<03:16,  5.80it/s]
[*] Training time: 89.29 s
[*] Testing the best model...
[*] Testing time: 0.03 s
[*] Final metrics:
[*] R2 mean: 0.6035
[*] R2 std: 0.0452
[*] RRMSE mean: 0.2203
[*] RRMSE std: 0.0164
[*] RMSES mean: 4.4569
[*] RMSES std: 0.6151
[*] RMSEU mean: 5.1047
[*] RMSEU std: 0.4252
[*] Training time mean: 74.96 s
[*] Training time std: 12.17 s
[*] Testing time mean: 0.03 s
[*] Testing time std: 0.00 s
```

The model's accuracy is automatically evaluated at the end of the training
process.

The following table shows the models tested in the research paper and the
appropriate commands to run them:

| Model           | Command                                                   |
|-----------------|-----------------------------------------------------------|
| MLP             | `guild run model=MLP warmup_epochs=1`                     |
| Bi-LSTM         | `guild run model=BiLSTM warmup_epochs=1`                  |
| Transformer     | `guild run model=Transformer warmup_epochs=1`             |
| FCN             | `guild run model=FCN warmup_epochs=1`                     |
| Residual CNN    | `guild run model=ResidualNet warmup_epochs=1`             |
| InceptionTime   | `guild run model=InceptionTime eps=0.01`                  |
| InceptionTime-5 | `guild run model=InceptionTime eps=0.01 ensemble_count=5` |
| ResNeTS         | `guild run model=ResNet18T`                               |
| ResNeTS-5       | `guild run model=ResNet18T ensemble_count=5`              |
| Rocket          | `guild run model=Rocket epochs=1 warmup_epochs=0`         |

The available study variables are: `Shannon`, `Simpson`, and `SpecRichness`.

## License

This project is licensed under the Apache License 2.0. See the
[LICENSE](LICENSE) file for details.

## Citation

If you find this work useful, please consider citing the corresponding research
paper:

```tex
@ARTICLE{10664042,
  author={Dieste, Álvaro G. and Argüello, Francisco and Heras, Dora B. and Magdon, Paul and Linstädter, Anja and Dubovyk, Olena and Muro, Javier},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={ResNeTS: a ResNet for Time Series Analysis of Sentinel-2 Data Applied to Grassland Plant-Biodiversity Prediction}, 
  year={2024},
  volume={},
  number={},
  pages={1-23},
  keywords={Time series analysis;Biodiversity;Remote sensing;Computer architecture;Grasslands;Long short term memory;Europe;biodiversity prediction;deep learning;multispectral imaging;remote sensing;residual network;sentinel-2;time series analysis},
  doi={10.1109/JSTARS.2024.3454271}
}
```
