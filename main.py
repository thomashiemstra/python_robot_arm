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

animation = animateArm(fig,ax)

position = np.array([-20,25,10])
pose = pose3D(position, True)
animation.draw_arm(pose, 'r')


animation2 = animateArm(fig,ax)

position = np.array([20,25,10])
pose = pose3D(position, True)
animation2.draw_arm(pose)

#animation.set_obstacles(obstacles)
#
#ani = animation.runAnimation()

