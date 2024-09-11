FROM nvcr.io/nvidia/pytorch:22.11-py3

ENV DEBIAN_FRONTEND=noninteractive
# Prevents Python from writing bytecode files during execution
ENV PYTHONDONTWRITEBYTECODE=1
# To allow deterministic behaviour when running PyTorch on CUDA
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

WORKDIR /opt/ResNeTS

RUN apt-get update
RUN apt-get install -y tmux

# To read Excel files with Pandas
RUN pip install openpyxl==3.0.10

# To report info about neural networks
RUN pip install torchinfo==1.7.1
RUN pip install fvcore==0.1.5.post20221221

# To automate, track, and compare experiments
RUN pip install guildai==0.8.2
RUN guild completion --install --shell bash
ENV GUILD_HOME=/opt/ResNeTS/.guildai

# Provides the Rocket model
RUN pip install sktime==0.15.1
