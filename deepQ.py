# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from simulation_utils import box, simulation
from kinematics import pose3D
from datetime import datetime




class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.90    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(36, input_dim=self.state_size, activation='relu'))
        model.add(Dense(36, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    EPISODES = 100000
    
    box = box([10,10,30], pos=[-5,20,0])   
    obstacles = np.array([box])
    
    position = np.array([-20,25,10])
    initial_pose = pose3D(position, True)
    
    position = np.array([20,25,10])
    target_pose = pose3D(position, True)
    
    state_size = 21
    action_size = 32
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-ddqn.h5")
    done = False
    batch_size = 50
    
    startTime = datetime.now()
    for e in range(EPISODES):
        sim = simulation(initial_pose, target_pose, obstacles, radius = 5)
        state = sim.reset()
        state = np.reshape(state, [1, state_size])
        score = 0
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = sim.step(action)
            score += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                if e % 10 == 0:
                    print("episode: {}/{}, score: {}, e: {:.2}"
                          .format(e, EPISODES, score, agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            if e % 1000 == 0:
                agent.save("./save/obstacle_avoidance-ddqn_episode_" + str(e) + "_score_" + str(score) +  "_.h5")
                
    print("it took:",datetime.now() - startTime)