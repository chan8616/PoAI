#!/bin/bash
conda create -n poai_gpu -c anaconda python=3.7 tensorflow==2.1 keras=2.3.1 scikit-learn pandas colorama -y
eval "$(conda shell.bash hook)"
conda activate poai_gpu
conda install -c conda-forge wxpython gooey==1.0.4 pycocotools Cython -y
conda install -y pytorch=1.5.1 torchvision cudatoolkit=10.1 -c pytorch
sudo apt-get install openjdk-8-jdk python-dev
pip install -U konlpy jpype1-py3 gensim