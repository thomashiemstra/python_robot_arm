from os import environ
environ['MPLBACKEND'] = 'module://gr.matplotlib.backend_gr'

from simulation_utils import box, plot_world, animateArm, simulation
import numpy as np
from kinematics import pose3D



fig, ax = plot_world()

box = box([10,10,30], pos=[-5,20,0])
box.plot(ax)

obstacles = np.array([box])


#
#animation = animateArm(fig,ax, radius = 3.0)
#
#animation.set_obstacles(obstacles)
#
#ani = animation.runAnimation()




def draw_initial_world(obstacles, fig, ax, initial_pose, target_pose, radius = 0.0):
    animation = animateArm(fig,ax, radius=radius)
    animation.set_obstacles(obstacles)
    animation.draw_arm(initial_pose)
    
    animation2 = animateArm(fig,ax, radius=radius)
    animation2.set_obstacles(obstacles)
    animation2.draw_arm(target_pose)

position = np.array([-12,25,10])
initial_pose = pose3D(position, True)

position = np.array([20,25,10])
target_pose = pose3D(position, True)
    
draw_initial_world(obstacles, fig, ax, initial_pose, target_pose, radius =3)


sim = simulation(initial_pose, target_pose, obstacles, radius = 3.0)

observation = sim.start()

np.set_printoptions(threshold=np.inf)
#print(observation)

#angles = [ 0.          2.0032041   1.24224632 -0.72598384 -2.65368372 -1.10604683 2.90807073]
