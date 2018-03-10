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

a = np.log(2)/25

apdataX = np.random.random((5, 35))
quarter_way_arr = [False, False, False]

quarter_way_arr[0] = True
quarter_way_arr[1] = True
quarter_way_arr[2] = True

mat = np.eye(3)
print(np.linalg.norm(mat))


