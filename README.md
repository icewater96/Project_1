# Project_1

This repository is Project 1 in Udacity Deep Reinforcement Learning. 

## Install dependencies

I have Windows 10 OS and did the following from the dependencies section of https://github.com/udacity/deep-reinforcement-learning:

conda create --name drlnd python=3.6

conda activate drlnd

pip install gym

pip install unityagents==0.4.0

pip install .

conda install pytorch=0.4.0 -c pytorch  # Some errors occurred when I ran the above line. My Undacity mentor suggested this line. 

## Training

Python code main.py in the repository contains all necessary code for this project. Running python main.py will start training automatically. The script will save intermediate model into checkpoint file. 

