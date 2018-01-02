from os import environ
environ['MPLBACKEND'] = 'module://gr.matplotlib.backend_gr'

from simulation_utils import box, plot_world, animateArm, simulation
import numpy as np
from kinematics import pose3D
import gym


def bitarray(n, base):
    temp = np.array([1 if digit=='1' else 0 for digit in bin(n)[2:]])
    res = np.zeros(base)
    res[0:temp.size] = temp
    return res
#        
#
#
#
#
#
#
fig, ax = plot_world()

box = box([10,10,30], pos=[-5,20,0])
box.plot(ax)

obstacles = np.array([box])



#animation = animateArm(fig,ax, radius = 3.0)
#
#animation.set_obstacles(obstacles)
#
#ani = animation.runAnimation()


#def draw_initial_world(obstacles, fig, ax, initial_pose, target_pose, radius = 0.0):
#    animation = animateArm(fig,ax, radius=radius)
#    animation.set_obstacles(obstacles)
#    animation.draw_arm(initial_pose)
#    
#    animation2 = animateArm(fig,ax, radius=radius)
#    animation2.set_obstacles(obstacles)
#    animation2.draw_arm(target_pose)
#
#
#

#

#
#animation = animateArm(fig,ax, radius=radius)
#animation.set_obstacles(obstacles)
#animation.draw_arm(initial_pose)

position = np.array([-20,25,10])
initial_pose = pose3D(position, True)


position = np.array([20,25,10])
target_pose = pose3D(position, True)
#    
#draw_initial_world(obstacles, fig, ax, initial_pose, target_pose, radius =4.0)

radius = 2

sim = simulation(initial_pose, target_pose, obstacles, radius = radius)
sim.setup_animation(fig,ax)
sim.draw_arm(initial_pose)

observation = sim.start()

print(observation.size)


np.set_printoptions(threshold=np.inf)

