# Project_1

This repository is Project 1 in Udacity Deep Reinforcement Learning. 

## Getting started

I have Windows 10 OS and did the following from the dependencies section of https://github.com/udacity/deep-reinforcement-learning:

conda create --name drlnd python=3.6

conda activate drlnd

pip install gym

pip install unityagents==0.4.0

pip install .

conda install pytorch=0.4.0 -c pytorch  # Some errors occurred when I ran the above line. My Undacity mentor suggested this line. 

Once the above steps are done, one needs to clone this repository to get the working code and saved model weights 

## How to run

Python code main.py in the repository contains all necessary code for this project. Train() in main.py is the entry function to train a model. Meanwhile, the main body of code is a complete workflow for training. Running python main.py will start training automatically. The script will save intermediate model into checkpoint file. 

## Learning algorithms

This implementation uses Deep Reinforcement Learning approach. To be more specific, Deep Q Networks (DQN) are utilized to represent a stata-action value function, converting a state vector into action values. The training process involves Experience Replay to break unwanted correlation in experiences. In addition, fixed Q target is also implemented for the sake of training stability. 

The Deep Q Networks consist of 3 dense layers. The first 2 layers have Relu activation and the last layer is just linear activated. The input layers is 37 dimensional and the output layer is 4 dimensional. Two hidden layers are both 64 dimensional. 

Other hyperparameters are listed below:
- BUFFER_SIZE = int(1e5)  # size of replay buffer
- BATCH_SIZE = 64   # minibatch size
- GAMMA = 0.99  # discount factor
- TAU = 1e-3    # for soft update of target parameters
- LR = 5e-4   # learning rate
- UPDATE_EVERY = 4   # how ofter to update the network

## Future work
This implement works fine but it takes a long time to train. I would like to try Prioritized Experience Replay and Dueling Network to improve performance. 
