# AI for the car
# importing the libs

import numpy as np
import random as rd # for random samples from batches
import os # load the module, save
import torch
import torch.nn as nn # implement neural networks
import torch.nn.functional as F # functions of neural networks
import torch.optim as optim # optimizer for stochastic gradient descent
import torch.autograd as autograd
from torch.autograd import Variable # convert tensors to variables with gradient

# the architecture of the neural network

class Network(nn.Module): #inherits from parent Module class
    
    def __init__(self, input_size, nb_action): # constructor of the NN
        super(Network, self).__init__()
        self.input_size = input_size # input_size is 5 encoded vector values
        self.nb_action = nb_action # nb_action is possible actions
        self.fc1 = nn.Linear(input_size, 30) # connection between input and hidden layer
        self.fc2 = nn.Linear(30, nb_action) # connection between hidden and output layer
    
    def forward(self, state): 
        x = F.relu(self.fc1(state)) # x represents hidden neurons, state is input neurons 
        q_values = self.fc2(x) # returns q_values
        return q_values
    
# Implement Experience Replay
    
class ReplayMemory(object): # holds current N transitions and doesnt allow more than 100
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, event): # event is a tuple of 4, 1st is old state, 2nd is new state, 3rd is last action, 4th is last reward
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
        
    def sample(self, batch_size): # batch size is how many samples will be taken
        samples = zip(*rd.sample(self.memory, batch_size)) # var to hold samples from the memory. Samples are tuples as (states), (actions) and (rewards)
        return map(lambda x: Variable(torch.cat(x, 0)), samples) # take samples concatanete them wrt first dimension and convert them to torch variables which holds both a tensor and a gradient to differentiate to update the weights
    
# Implement Deep Q-Learning
        
class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = [] # mean of 100 rewards, to see if it increases which means the network is improving
        self.model = Network(input_size, nb_action) # initialize Neural Network
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001) # lr should be small to give the agent time to learn from its mistakes with punishments
        self.last_state = torch.Tensor(input_size).unsqueeze(0) # 3 signals, orientation, - orientation, converted to tensor with batch dimentsion
        self.last_action = 0 # actions are represented as 0, 1 or 2. Dont need any tensor
        self.last_reward = 0 # reward can be -1 to 1, again no tensor is needed
    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True)) * 7) # for probs of q-values the NN is called, state will be given but model accepts tensors so we convert via Variable(), volatile means it wont add gradient to the tensor, Temperature(T = 7) means how sure the NN will be sure to play the action it picks, closer to 0 -> less sure 
        # softmax([1, 2, 3]) = [0.04, 0.11, 0.85] => softmax([1, 2, 3] * 3) = [0, 0.02, 0.98] , now the action with the highest prob is more likely to be chosen
        '''
        probs -> tensor([[1.9094e-15, 9.8416e-01, 1.5845e-02]], grad_fn=<SoftmaxBackward>)
        '''
        action = probs.multinomial(1) # we pick an action randomly
        '''
        action -> tensor([[2]])
        action.data[0, 0] -> tensor(2)
        '''
        return action.data[0, 0] # action is returned with fake batch, similar to first dimension 
        
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action): # parameters are transition from Markov Decision Process: batches of current_ state, next_state, reward, action
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1) # returns only the action chosen for each input state of batch
        
        '''
        outputs -> tensor([-0.7965, -0.9505, -0.7247, -1.0051, -1.7359, -0.9409, -0.9344, -0.5163,
        -0.9176, -0.9491, -0.7546, -0.9514, -0.9523, -0.8359, -0.7408, -0.9519,
        -0.9963, -0.9927, -0.9726, -0.9707, -0.9311, -0.9591, -0.8849, -0.9692,
        -0.9738, -0.9565, -1.0279, -1.7371, -0.9588, -1.0901, -0.9078, -0.9664,
        -0.9598, -0.7590, -0.9829, -0.5814, -0.7894, -0.9502, -0.9605, -0.8246,
        -0.9715, -1.7492, -0.9528, -0.7568, -0.7340, -0.9685, -0.9787, -0.9877,
        -1.7854, -0.9859, -1.5047, -1.0144, -0.9672, -0.9699, -1.0350, -0.9947,
        -0.9250, -0.7870, -1.7377, -1.0437, -0.7819, -0.9466, -0.9894, -1.0739,
        -1.0901, -0.9658, -0.9649, -0.6449, -2.3237, -0.7389, -1.0092, -0.8452,
        -0.9821, -0.7106, -0.5163, -1.0418, -0.9955, -2.3237, -1.0105, -0.8986,
        -0.7265, -1.0090, -0.8414, -0.9491, -0.8314, -1.0319, -1.0748, -0.7972,
        -0.5873, -0.9561, -1.0044, -0.9450, -1.0194, -1.5121, -0.7070, -0.7044,
        -0.9918, -0.9694, -1.2182, -1.0154]
        '''
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        
        '''next_outputs -> torch.Size([100])
        
        next_outputs -> tensor([-4.6006e-02,  3.6891e-01, -9.8510e-01, -1.1861e-02, -7.2663e-01,
        -5.9830e-01, -7.3048e-01, -1.0221e+00,  4.0348e-01, -7.4782e-01,
         4.4741e-01,  3.6605e-01, -4.6654e-01, -7.4748e-01,  9.3445e-02,
         4.4690e-01,  2.7304e-01,  3.7089e-01, -8.1604e-01,  1.3489e-01,
         7.2838e-02, -1.3189e-01, -2.4508e-01,  4.4744e-01, -3.3549e-01,
        -1.7982e-01, -2.0913e-01, -1.2712e+00,  4.3502e-01, -7.4815e-01,
        -7.4826e-01, -3.1748e-01, -8.1775e-01, -8.0851e-01,  4.4696e-01,
        -6.2519e-02,  1.4528e-01, -1.5231e-01, -5.9384e-01, -7.3961e-01,
        -3.2621e-01,  3.7961e-01, -7.4829e-01,  3.9704e-01,  1.1560e-01,
        -2.8727e-03,  2.0515e-01,  9.3352e-02, -1.9356e-01, -1.5749e+00,
         1.2996e-02,  2.3187e-01,  1.0106e-01, -2.4116e+00, -5.4394e-01,
        -5.3046e-02, -8.1692e-01, -3.3150e-01,  2.8527e-01,  1.2291e-01,
         1.1870e-01,  4.0098e-01,  3.7733e-01, -5.7901e-01,  1.7269e-01,
         2.8657e-01,  3.0695e-01,  3.7626e-01, -1.8999e-01,  3.9884e-01,
        -8.0935e-01, -3.3285e-01,  1.2847e-01, -1.2343e-01, -5.1729e-02,
        -3.6648e-01,  3.0203e-01, -2.4205e+00, -4.3673e-02,  3.9634e-01,
         1.7572e-02,  1.9069e-01, -7.2499e-01, -7.1445e-01,  8.3707e-02,
         1.0204e-01,  2.2102e-01, -3.3544e-01, -5.4675e-01,  4.4748e-01,
         2.8439e-01, -3.3524e-01, -7.2307e-01, -1.6450e-01, -6.7772e-04,
         2.4186e-01,  5.9342e-02,  2.6569e-01,  2.7154e-01, -4.7149e-01])
        '''
        target = self.gamma*next_outputs + batch_reward
        print('target size:')
        print(target.size())
        ''' 
        target size -> torch.Size([100])
        
        target -> tensor([-2.1167, -1.5119, -3.1451, -1.8496, -1.5644, -2.3905, -3.1450, -0.9643,
        -1.5879, -1.1577, -2.6802, -0.9752, -3.1450, -3.4560, -0.9642, -1.1586,
        -0.9760, -1.5888, -3.4566, -1.4749, -2.6806, -1.0416, -1.4169, -2.3950,
        -1.6515, -3.4553, -2.6840, -1.5883, -1.5652, -2.2238, -1.0090, -0.9641,
        -1.0453, -2.6547, -1.5882, -0.9740, -1.5731, -1.0441, -2.6559, -1.5137,
        -1.1046, -2.3646, -2.3512, -1.5882, -3.4565, -1.3551, -0.9643, -1.5885,
        -1.5884, -1.6434, -1.5883, -0.9764, -3.4553, -3.1450, -1.4186, -1.4131,
        -2.3636, -1.5699, -0.9644, -3.1451, -1.5887, -2.1070, -2.6822, -1.6441,
        -1.5884, -1.1021, -1.6429, -1.0097, -0.9754, -2.6829, -0.9760, -1.5114,
        -1.0431, -1.5885, -1.0085, -1.3151, -1.6444, -1.5885, -1.3134, -3.1450,
        -3.1450, -2.6766, -1.4792, -1.5631, -3.4558, -1.6440, -1.3126, -2.6778,
        -2.3052, -1.5077, -1.5649, -2.3531, -1.5654, -1.5218, -2.0428, -1.5878,
        -1.5058, -1.5083, -1.6432, -3.4561])
        '''
        td_loss = F.smooth_l1_loss(outputs, target) # Temporal Difference, mean absolute error of the elements in each tensor, divided by their length
        '''
        td_loss -> tensor(0.0906, grad_fn=<SmoothL1LossBackward>)
        '''
        self.optimizer.zero_grad() # Backpropagate the network with SGD, reinitialize the optimizer for each iteration in the loop of SGD. Clears the gradients of all optimized tensor
        td_loss.backward(retain_graph = True) # backpropagate and to improve the performance set True to free memory, useful since learning isnt cheap, computes loss/dx for each parameters() value
        self.optimizer.step() # update the weights of the neural network
        #params = list(self.model.parameters())
        #print(params)
    
    def update(self, reward, new_signal): # when reaching a new state update the new reward and signal and return the action
        new_state = torch.Tensor(new_signal).float().unsqueeze(0) # define new state as a Tensor and convert type to float for more accuracy, unsqueeze to show fake dimension for Tensor
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]))) # we only dont have batch_next_state in learn, we need to add it to the memory, last_action is Tensor since it will be feed to the NN
        action = self.select_action(new_state) # now play an action with the new state 
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100) # learn from 100 transitions from memory
            '''batch_next_state 5 element as ex -> tensor([[ 0.0000,  0.0000,  0.0000,  0.3850, -0.3850],
                                                   [ 0.0000,  1.0000,  0.0000,  0.4466, -0.4466],
                                                   [ 0.0000,  0.0000,  0.0000,  0.1226, -0.1226],     
                                                   [ 0.0000,  0.0000,  0.0000,  0.9264, -0.9264],
                                                   [ 0.0000,  0.0000,  0.0000, -0.4209,  0.4209]]) 
    
            batch_action -> tensor([2, 0, 1, 0, 0, 0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                                    0, 1, 1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 2, 0, 1,
                                    0, 0, 1, 0, 2, 0, 1, 0, 2, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 2, 0, 0, 1, 0,
                                    0, 1, 0, 2, 1, 1, 0, 0, 0, 1, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
                                    0, 0, 1, 2])
    
            batch_reward -> tensor([ 0.1000,  0.1000, -0.2000,  0.1000,  0.1000, -0.2000,  0.1000,  0.1000,
                                    0.1000, -0.2000,  0.1000,  0.1000,  0.1000, -0.2000,  0.1000,  0.1000,
                                    0.1000,  0.1000,  0.1000, -0.2000,  0.1000,  0.1000,  0.1000, -0.2000,
                                    -0.2000,  0.1000,  0.1000,  0.1000,  0.1000, -0.2000,  0.1000,  0.1000,
                                    -1.0000,  0.1000,  0.1000,  0.1000,  0.1000,  0.1000,  0.1000,  0.1000,
                                    0.1000,  0.1000,  0.1000,  0.1000,  0.1000, -0.2000,  0.1000, -0.2000,
                                    0.1000,  0.1000, -0.2000,  0.1000,  0.1000,  0.1000,  0.1000, -0.2000,
                                    -0.2000, -1.0000, -1.0000,  0.1000, -0.2000, -1.0000,  0.1000,  0.1000,
                                    -0.2000, -1.0000,  0.1000,  0.1000,  0.1000,  0.1000,  0.1000,  0.1000,
                                    -0.2000,  0.1000, -1.0000,  0.1000,  0.1000,  0.1000,  0.1000, -0.2000,
                                    0.1000, -1.0000, -1.0000,  0.1000,  0.1000, -1.0000,  0.1000,  0.1000,
                                    -0.2000,  0.1000,  0.1000,  0.1000,  0.1000,  0.1000, -0.2000,  0.1000,
                                    0.1000, -0.2000,  0.1000,  0.1000])
        '''
            self.learn(batch_state, batch_next_state, batch_reward, batch_action) # the agent learns from 100 states, 100 next_states, 100 rewards, 100 actions
        self.last_action = action # update the last action after playing
        self.last_state = new_state # update the last state after playing
        self.last_reward = reward # update the last reward after playing
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000: # keep track of reward mean
            del self.reward_window[0]
        return action
    
    def score(self): # returns mean of reward_window
        return sum(self.reward_window)/(len(self.reward_window)+1) # +1 is to avoid sum/0 error
        
    def save(self): # save the last weights of the last iteration and the optimizer
        torch.save({'state_dict' : self.model.state_dict(), 'optimizer' : self.optimizer.state_dict()}, 'last_brain.pth') # weights and optimizer will be saved to last_brain.pth when the function is called
        
    def load(self):
        if os.path.isfile('last_brain.pth'): # checks if the file exists
            print("=> loading checkpoint")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Done")
        else:
            print("No checkpoint found...")
        
        
        
        