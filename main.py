from os import environ
environ['MPLBACKEND'] = 'module://gr.matplotlib.backend_gr'

from simulation_utils import box, plot_world, animateArm, simulation
import numpy as np
from kinematics import pose3D
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

state_size = 21
action_size = 32
learning_rate = 0.001

def huber_loss(target, prediction):
    # sqrt(1+error^2)-1
    error = prediction - target
    return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

def build_model():
    # Neural Net for Deep-Q learning Model
    model = Sequential()
    model.add(Dense(36, input_dim=state_size, activation='relu'))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss=huber_loss,
                  optimizer=Adam(lr=learning_rate))
    return model



model = build_model()
name = "./save/obstacle_avoidance-ddqn_episode_99000_score_-88.6969851585_cut_off_5_.h5"
model.load_weights(name)


fig, ax = plot_world()

box = box([10,10,30], pos=[-5,20,0])
box.plot(ax)

obstacles = np.array([box])


position = np.array([-20,25,10])
initial_pose = pose3D(position, True)

position = np.array([20,25,10])
target_pose = pose3D(position, True)


sim = simulation(initial_pose, target_pose, obstacles, radius = 5)
#sim.setup_animation(fig,ax)
#sim.draw_arm(initial_pose)

angles = sim.generate_animation_angles(model)

#print(angles)
steps = angles.shape[0]



animation = animateArm(fig,ax)
animation.set_animation_angles(angles, steps)

an = animation.runAnimation_angles()



