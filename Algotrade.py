'''
predicting the stock prices with the help of reinforcement learning
this tutorial goes in step by step procedure and uses q learning
data used: from yahoo finance
author : Bharath Kumar (bharathjatoth.github.io)
Step 1 : import all the libraries which are used by the algo
'''
import keras
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizer_v2.adam import Adam
import math
import random,sys
from collections import deque

#creating a class agent which has all the static data threshold values
class Agent:
    def __init__(self,state_size,is_eval=False,model_name=""):
        self.state_size = state_size
        self.action_size = 3  #buy sell and hold
        self.memory = deque(maxlen=100)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.model = load_model(model_name) if is_eval else self._model()

    def _model(self):
        model = Sequential()
        model.add(Dense(units=64,input_dim=self.state_size,activation='relu'))
        model.add(Dense(units=32,activation='relu'))
        model.add(Dense(units=8,activation="relu"))
        model.add(Dense(self.action_size,activation="linear"))
        model.compile(loss='mse',optimizer=Adam(lr=0.0001))
        return model

    def act(self,state):
        if not self.is_eval and random.random()<=self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(state)
        return np.argmax(options[0])

    def expreplay(self,batch_size):
        minibatch = []
        l = len(self.memory)
        for i in range(1-batch_size+1,1): minibatch.append(self.memory[i])
        for state, action,reward, next_state,done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma*np.argmax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state,target_f,epochs=1,verbose=0)
        if self.epsilon > self.epsilon_min: self.epsilon *=self.epsilon_decay

def format_price(n):
    return ("-Rs. " if n<0 else "Rs."+"{:.2f}%".format(abs(n)))

def getstockdatavec(key):
    vec = []
    lines = open(r'filename to be input from here','r').read().splitlines()
    for line in lines[1:]:
        if line.split(",")[4] != "null":
            vec.append(float(line.split(",")[4]))
    return vec

def sigmoid(x):
    return 1/(1+math.exp(-x))

def getstate(data,t,n):
    d = t-n+1
    block = data[d:t+1] if d>=0 else -d*[data[0]] + data[0:t+1]
    res = []
    for i in range(n-1):
        res.append(sigmoid(block[i+1]-block[i]))
    return np.array([res])

#training the agent
stock_name = input("Enter the stock name, window_size, episode_count : ")
window_size = input()
episode_count = input()
stock_name = str(stock_name)
window_size = int(window_size)
episode_count = int(episode_count)
agent = Agent(window_size)
data = getstockdatavec(stock_name)
l = len(data) - 1
batch_size = 32
for episode in range(episode_count+1):
    print("episode number : ",episode)
    state = getstate(data,0,window_size+1)
    total_profit = 0
    agent.inventory = []
    for t in range(l):
        print(state)
        action = agent.act(state)
        next_state = getstate(data,t+1,window_size+1)
        reward = 0
        if action==1: #buy
            agent.inventory.append(data[t])
            print('buy : ',format_price(data[t]))
        elif action==2 and len(agent.inventory) > 0:
            #sell the share
            bought_price = window_size_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price,0)
            total_profit += data[t]-bought_price
            print("profit : ",format_price(data[t]-bought_price))
        done=True if t == l-1 else False
        agent.memory.append((state,action,reward,next_state,done))
        state = next_state
        if done:
            print('total profit ---------------',format_price(total_profit))
        if len(agent.memory) > batch_size:
            agent.expreplay(batch_size)
    if episode % 10 == 0:
        agent.model.save(str(episode))

