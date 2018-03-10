from simulation_utils import box, plot_world, animateArm, simulation
import numpy as np
from kinematics import pose3D
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers


from os import environ
environ['MPLBACKEND'] = 'module://gr.matplotlib.backend_gr'

num_env_variables = 30
action_size = 5
radius = 4.0
cut_off = 3

def build_model(num_env_variables, num_env_actions):
    # initialize the action predictor model
    action_predictor_model = Sequential()
    # model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
    action_predictor_model.add(Dense(256, activation='relu', input_dim=num_env_variables))
    action_predictor_model.add(Dropout(0.5))
    action_predictor_model.add(Dense(256, activation='relu'))
    action_predictor_model.add(Dropout(0.5))
    action_predictor_model.add(Dense(256, activation='relu'))
    action_predictor_model.add(Dropout(0.2))
    action_predictor_model.add(Dense(num_env_actions))
    opt2 = optimizers.adam()
    action_predictor_model.compile(loss='mse', optimizer=opt2, metrics=['accuracy'])
    return action_predictor_model


model = build_model(num_env_variables, action_size)
weigths_filename = "obstacle-avoidance-v1-weights-ap.h5"
model.load_weights(weigths_filename)


fig, ax = plot_world()

box1 = box([6, 20, 15], pos=[-3, 10, 0])
box2 = box([10, 10, 40], pos=[-5, 25, 0])
obstacles = np.array([box1, box2])

position = np.array([-15, 25, 10])
initial_pose = pose3D(position, False)

position = np.array([15, 25, 10])
target_pose = pose3D(position, False)

box1.plot(ax)
box2.plot(ax)


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
