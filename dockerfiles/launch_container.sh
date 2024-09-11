#!/usr/bin/env bash

docker run \
    -it \
    --gpus all \
    --shm-size=1g \
    --rm \
    -p 6006:6006 \
    -v /home/alvaro.goldar/ResNeTS:/opt/ResNeTS \
    resnets
