# Project_1

This repository is Project 1 in Udacity Deep Reinforcement Learning. 

## Getting started

The framework is based on Pytorch and Python 3.6. 

My implementation is based on Windows 10 OS and did the following from the dependencies section of https://github.com/udacity/deep-reinforcement-learning:

conda create --name drlnd python=3.6

conda activate drlnd

pip install gym

pip install unityagents==0.4.0

pip install .

conda install pytorch=0.4.0 -c pytorch  # Some errors occurred when I ran the above line. My Undacity mentor suggested this line. 

Once the above steps are done, one needs to clone this repository to get the working code and saved model weights.

## How to run

Python code main.py in the repository contains all necessary code for this project. Train() in main.py is the entry function to train a model. Meanwhile, the main body of code is a complete workflow for training. Running python main.py will start training automatically. The script will save intermediate model into checkpoint file. 

Serveral pretrained models have been already saved in this repository as .pth files. 
