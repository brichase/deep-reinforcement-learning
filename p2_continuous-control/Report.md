# Continuous Control Project Report

### Learning Algorithm: DDPG

The four action vectors in this environment (corresponding to the torque applicable to the two joints) are **continuous values between -1 and 1.**

For the continuous action space, the learning algorithm I used is **the Deep Deterministic Policy Gradient Algorithm - DDPG**. This algorithm is similar to Deep-Q Network in that there is a target network that is guiding training the policy. However, DDPG being a policy-based method, the algorithm search directly for the optimal policy without simultaneously maintaining a value function estimate needing a discrete action space.

##### Hyperparameters
The hyperparameters used to define the DDPG algorithm to solve this environment closely resembles what is used in the Udacity deep-reinforcement-learning github repo for the openai pendulum environment.

What I found helped the algorithm perform well is changing the learning rate of the critic to be the same as the actor at 1e-4. When the learning rate for the critic was set to 1e-3, the algorithm did not learn to solve the environment and score plateaued before hitting the average score of 30.  

```python
# Hyperparmeters Used
BUFFER_SIZE = int(1e5)  # replay buffer size  
BATCH_SIZE = 128        # minibatch size  
GAMMA = 0.99            # discount factor  
TAU = 1e-3              # for soft update of target parameters  
LR_ACTOR = 1e-4         # learning rate of the actor  
LR_CRITIC = 1e-4        # learning rate of the critic  
WEIGHT_DECAY = 0        # L2 weight decay  
```

##### Neural Net Model Architecture
The actor and critic both use two full-connected layers with a relu activation function on each layer. The first FC layer comprises of 400 nodes and the second of 300 nodes. The actor output undergoes the tanh activation function which keeps the output of the network between -1 and 1.

Ornstein-Uhlenbeck noise is applied to action to push the agent to explore the space. The parameters used are mu=0., theta=0.15, sigma=0.2 for the OUnoise.

### Plot of Rewards
Episode 100     Average Score: 25.03
Episode 114     Average Score: 30.04
Environment solved in 114 episodes!     Average Score: 30.04

![Plot of Rewards](plot_of_rewards.png)

### Ideas for Future Work
Benchmark different algorithms like vanilla PPO and actor-critic algorithms like A3C, A2C, GAE.

![action vs parameter noise](action_vs_parameter_noise.png)

[OpenAI has a blog post](https://blog.openai.com/better-exploration-with-parameter-noise/) showing that adding noise to the parameters instead of the resulting action. I would like to implement the change and benchmark the performance to see what difference it makes in this environment.


### Learning Algorithm

The four action vectors in this environment (corresponding to the torque applicable to the two joints) are **continuous values between -1 and 1.**

For the continuous action space, the learning algorithm I used is **DDPG - the Deep Deterministic Policy Gradient Algorithm**. This algorithm is similar to Deep-Q Network in that there is a target network that is guiding training the policy. However, DDPG being a policy-based method, the algorithm search directly for the optimal policy without simultaneously maintaining a value function estimate needing a discrete action space.


Explain here the hyperparameters

Describe the model architectures for the neural networks.

### Plot of Rewards
Episode 100	Average Score: 25.03  
Episode 114	Average Score: 30.04  
Environment solved in 114 episodes!	Average Score: 30.04  

![Plot of Rewards](plot_of_rewards.png)

### Ideas for Future Work
Benchmark different algorithms like vanilla PPO and actor-critic algorithms like A3C, A2C, GAE.

![action vs parameter noise](action_vs_parameter_noise.png)

[OpenAI has a blog post](https://blog.openai.com/better-exploration-with-parameter-noise/) showing that adding noise to the parameters instead of the resulting action. I would like to implement the change and benchmark the performance to see what difference it makes in this environment.
