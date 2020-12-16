#!/bin/bash
conda create -n poai python=3.7 tensorflow==2.3.0 keras=2.3.1 scikit-learn matplotlib tensorboard pandas colorama -y && \
eval "$(conda shell.bash hook)" && \
conda activate poai && \
conda install -c conda-forge gooey pycocotools Cython -y && \
conda install pytorch=1.5.1 torchvision cpuonly -c pytorch && \
sudo apt-get install openjdk-8-jdk python-dev && \
pip install -U konlpy jpype1-py3 gensim