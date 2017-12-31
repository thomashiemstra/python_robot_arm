from os import environ
environ['MPLBACKEND'] = 'module://gr.matplotlib.backend_gr'

from simulation_utils import box, plot_world, animateArm
import numpy as np
from kinematics import pose3D



fig, ax = plot_world()

box1 = box([2,35,15], pos=[-10,0,0]) 
box2 = box([2,35,15], pos=[10,0,0])  

box1.plot(ax)
box2.plot(ax)

obstacles = np.array([box1,box2])


def draw_initial_world(obstacles, fig, ax, initial_pose, target_pose):
    animation = animateArm(fig,ax)
    animation.set_obstacles(obstacles)
    animation.draw_arm(initial_pose)
    
    animation2 = animateArm(fig,ax)
    animation2.set_obstacles(obstacles)
    animation2.draw_arm(target_pose)

position = np.array([-20,25,10])
initial_pose = pose3D(position, True)

position = np.array([20,25,10])
target_pose = pose3D(position, True)
    
draw_initial_world(obstacles, fig, ax, initial_pose, target_pose)

#animation.set_obstacles(obstacles)
#
#ani = animation.runAnimation()

