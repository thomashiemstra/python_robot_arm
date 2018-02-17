# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from simulation_utils import box, simulation
from kinematics import pose3D


a = np.array([-5,4,0])

