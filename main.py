from os import environ
environ['MPLBACKEND'] = 'module://gr.matplotlib.backend_gr'

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos, pi, sqrt

#fig = plt.figure()
#ax = Axes3D(fig)
#ax.set_xlim(-10, 10)
#ax.set_ylim(0, 20)
#ax.set_zlim(0, 20)
#ax.set_xlabel('X ---->')
#ax.set_ylabel('Y ---->')
#ax.set_zlabel('Z ---->')

class box:
    size = np.zeros(3)
    pos = np.zeros(3)
    
    def __init__(self, size, pos, angle):
        self.size = size
        self.pos = pos
        self.angle = angle

    def cuboid_data(self):
        # code taken from
        # https://stackoverflow.com/a/35978146/4124317 
        pos = self.pos
        for i in range(0,3):
            pos[i] += self.size[i]/2
            
        # suppose axis direction: x: to left; y: to inside; z: to upper
        # get the (left, outside, bottom) point
        o = [a - b / 2 for a, b in zip(pos, self.size)]
        # get the length, width, and height
        L, W, h = self.size
        
        Ca = cos(self.angle)
        Sa = sin(self.angle)
        
        x = np.matrix([[o[0], o[0] + L*Ca, o[0] + (L*Ca - W*Sa), o[0] - W*Sa, o[0]],                # x coordinate of points in bottom surface
                       [o[0], o[0] + L*Ca, o[0] + (L*Ca - W*Sa), o[0] - W*Sa, o[0]],                # x coordinate of points in upper surface
                       [o[0], o[0] + L*Ca, o[0] + L*Ca, o[0], o[0]],                # x coordinate of points in outside surface   
                       [o[0] - W*Sa, o[0] + (L*Ca - W*Sa), o[0] + (L*Ca - W*Sa), o[0] - W*Sa, o[0] - W*Sa]]  )             # x coordinate of points in inside surface
        
        y = np.matrix([[o[1], o[1] + L*Sa, o[1] + (L*Sa + W*Ca), o[1] + W*Ca, o[1]],      # y coordinate of points in bottom surface
                       [o[1], o[1] + L*Sa, o[1] + (L*Sa + W*Ca), o[1] + W*Ca, o[1]],                # y coordinate of points in upper surface
                       [o[1], o[1] + L*Sa, o[1] + L*Sa, o[1], o[1]],                        # y coordinate of points in outside surface
                       [o[1] + W*Ca, o[1] + (L*Sa + W*Ca), o[1] + (L*Sa + W*Ca), o[1] + W*Ca, o[1] + W*Ca]])   # y coordinate of points in inside surface
        
        z = np.matrix([[o[2], o[2], o[2], o[2], o[2]],                       
                       [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],   
                       [o[2], o[2], o[2] + h, o[2] + h, o[2]],               
                       [o[2], o[2], o[2] + h, o[2] + h, o[2]]]    )
        
        return x, y, z

    def plot(self,ax,color = 'b'):
        X, Y, Z = self.cuboid_data()
        ax.plot_surface(X, Y, Z, color=color, rstride=1, cstride=1, alpha=1)
        
#    vector of the point expressed in the cube's coordinate frame
    def getRPrime(self, r_point):
        c, s = np.cos(self.angle), np.sin(self.angle)
        R = np.array([[c, -s, 0], [s, c, 0],[0, 0, 1]])
        
        r_temp = np.zeros(3)
        for i in range(0,3):
            r_temp[i] = r_point[i] - self.pos[i]
        
        R = R.transpose()
                
        
        return np.asarray(R.dot(r_temp))
    
        
    def checkColission(self, r_point, radius=0.0):
        r_prime = self.getRPrime(r_point)
        
        x = r_prime[0]
        y = r_prime[1]
        z = r_prime[2]
        
        
        print(x,y,z)
        L, W, H = self.size
        
        if x < 0:
            dx = np.maximum(-x + radius, 0) #dx > 0 means collision due to radius of point
        elif x > L:
            dx = np.minimum(L - x + radius,0) #dx < 0 means collision due to radius of point
        else:
            dx = 0
            
        if y < 0:
            dy = np.maximum(-y + radius, 0) #dx > 0 means collision due to radius of point
        elif y > W:
            dy = np.minimum(L - y + radius,0) #dx < 0 means collision due to radius of point
        else:
            dy = 0
            
        if z < 0:
            dz = np.maximum(-z + radius, 0) #dx > 0 means collision due to radius of point
        elif z > H:
            dz = np.minimum(L - z + radius,0) #dx < 0 means collision due to radius of point
        else:
            dz = 0
        
        r = sqrt(dx*dx + dy*dy + dz*dz)
        
        return r


        
cube = box([5,5,5], [5,5,0], 0)
r = cube.checkColission([7,6,0])
print(r)
#cube.plot(ax)
