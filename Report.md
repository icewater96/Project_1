# Report

## Learning algorithms

This implementation uses Deep Reinforcement Learning approach. To be more specific, Deep Q Networks (DQN) are utilized to represent a stata-action value function, converting a state vector into action values. The training process involves Experience Replay to break unwanted correlation in experiences. In addition, fixed Q target is also implemented for the sake of training stability. 

The Deep Q Networks consist of 3 dense layers. The first 2 layers have Relu activation and the last layer is just linearly activated. The input layers is 37 dimensional and the output layer is 4 dimensional. Two hidden layers are both 64 dimensional. 

Other hyperparameters are listed below:
- BUFFER_SIZE = int(1e5)  # size of replay buffer
- BATCH_SIZE = 64   # minibatch size
- GAMMA = 0.99  # discount factor
- TAU = 1e-3    # for soft update of target parameters
- LR = 5e-4   # learning rate
- UPDATE_EVERY = 4   # how ofter to update the network

## Plot of rewards
This plot demonstrates the training process. The Y-Axis is average of reward over 100 episodes. As the training goes on, the average reward over 100 episode reached above 16.0. 

![](/Figure_1.png)


## Future work
This implement works fine but it takes a long time to train. I would like to try Prioritized Experience Replay and Dueling Network to improve performance. 
