from simulation_utils import box, plot_world, animateArm, simulation
import numpy as np
from kinematics import pose3D
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

from os import environ
environ['MPLBACKEND'] = 'module://gr.matplotlib.backend_gr'

state_size = 30
action_size = 5
radius = 2.0

def build_model(num_env_variables, num_env_actions):
    # initialize the action predictor model
    action_predictor_model = Sequential()
    # model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
    action_predictor_model.add(Dense(4096, activation='tanh', input_dim = num_env_variables))
    action_predictor_model.add(Dense(num_env_actions))
    return action_predictor_model


model = build_model(state_size, action_size)
weigths_filename = "obstacle-avoidance-v1-weights-ap.h5"
model.load_weights(weigths_filename)


fig, ax = plot_world()


box = box([10, 10, 40], pos=[-5, 25, 0])
obstacles = np.array([box])

position = np.array([-10, 25, 10])
initial_pose = pose3D(position, True)

position = np.array([10, 25, 10])
target_pose = pose3D(position, True)

box.plot(ax)



cut_off = 5
sim = simulation(initial_pose, target_pose, obstacles, radius=radius, cut_off=cut_off)
sim.reset()
#sim.setup_animation(fig,ax)
#sim.draw_arm(initial_pose)


angles, score = sim.generate_animation_angles(model)

#print(angles)
steps = angles.shape[0]
print(steps)
print("----------------------------------")

animation = animateArm(fig, ax, radius=radius)
animation.set_animation_angles(angles, steps)
animation.set_obstacles(obstacles)

#animation.animate_angles(steps-1)
an = animation.runAnimation_angles()
