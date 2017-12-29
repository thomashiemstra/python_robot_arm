import numpy as np
from numpy import sin, cos, sqrt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from kinematics import pose3D,inverseKinematics,forwardPosKinematics,forwardKinematicsRotation,d6

class box:
    size = np.zeros(3)
    pos = np.zeros(3)
    colission = False
    
    def __init__(self, size, pos=[0.0,0.0,0.0], angle=0.0, color='b'):
        self.size = size
        self.pos = pos
        self.angle = angle
        self.color = color

    def setColor(self,color):
        self.color = color

    def cuboid_data(self):
        # code taken from
        # https://stackoverflow.com/a/35978146/4124317 
#        pos = self.pos
        pos = np.zeros(3)
        for i in range(0,3):
            pos[i] =  self.pos[i] + self.size[i]/2
            
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

    def plot(self,ax):
        X, Y, Z = self.cuboid_data()
        ax.plot_surface(X, Y, Z, color=self.color, rstride=1, cstride=1, alpha=1)
        
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
        if r == 0:
            self.colission = True
        
        return r


def plotWorld():
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim(-20, 20)
    ax.set_ylim(0, 40)
    ax.set_zlim(0, 40)
    len = 10
    ax.plot([0,len], [0,0], [0,0], 'o-', lw=2, color = 'r')
    ax.plot([0,0], [0,len], [0,0], 'o-', lw=2, color = 'g')
    ax.plot([0,0], [0,0], [0,len], 'o-', lw=2, color = 'b')
    ax.set_xlabel('X ---->')
    ax.set_ylabel('Y ---->')
    ax.set_zlabel('Z ---->')
    return fig, ax


class animateArm:

#    def __init__(self, start_pose, stop_pose, ax, fig, steps=100, radius=5):
#        self.ax = ax
#        self.fig = fig
#        self.start = start_pose
#        self.stop = stop_pose
#        self.arm, = self.ax.plot([], [], [], 'yo-', lw=2)
#        self.gripL, = self.ax.plot([], [], [], 'yo-', lw=2)
#        self.gripR, = self.ax.plot([], [], [], 'yo-', lw=2)
#        self.lines = [self.arm,self.gripL,self.gripR]
#        self.steps = steps
#        self.radius=radius
    obstacles = np.array([])
        
    def __init__(self,fig,ax):
        self.ax = ax
        self.fig = fig
        self.arm, = self.ax.plot([], [], [], 'yo-', lw=2)
        self.gripL, = self.ax.plot([], [], [], 'yo-', lw=2)
        self.gripR, = self.ax.plot([], [], [], 'yo-', lw=2)
        self.lines = [self.arm,self.gripL,self.gripR]
        self.radius = 5
        
    def setObstacles(self, obstacles):
        self.obstacles = obstacles
        
    def getGripper(self, rot):
        temp = np.array([0,0,d6/2], dtype = np.float64)
        g00 = rot.dot(temp)
        temp = np.array([0,-2,0], dtype = np.float64)
        g11 = rot.dot(temp) 
        temp = np.array([0,-2,d6/2], dtype = np.float64)
        g12 = rot.dot(temp) 
        temp = np.array([0,2,0], dtype = np.float64)
        g21 = rot.dot(temp) 
        temp = np.array([0,2,d6/2], dtype = np.float64)
        g22 = rot.dot(temp)
        
        return g00,g11,g12,g21,g22
    
    def setArm(self, p1, p2, p4, g00, g11, g12, g21, g22):
        p5 = p4 + g00
        
        thisx = [0,p1[0], p2[0], p4[0], p5[0]]
        thisy = [0,p1[1], p2[1], p4[1], p5[1]]
        thisz = [0,p1[2], p2[2], p4[2], p5[2]]
        
        gripRx = [p5[0], p5[0]+g11[0], p5[0]+g12[0]]
        gripRy = [p5[1], p5[1]+g11[1], p5[1]+g12[1]]
        gripRz = [p5[2], p5[2]+g11[2], p5[2]+g12[2]]
        
        gripLx = [p5[0], p5[0]+g21[0], p5[0]+g22[0]]
        gripLy = [p5[1], p5[1]+g21[1], p5[1]+g22[1]]
        gripLz = [p5[2], p5[2]+g21[2], p5[2]+g22[2]]
     
        self.arm.set_data(thisx, thisy)
        self.arm.set_3d_properties(thisz)
        
        self.gripR.set_data(gripRx, gripRy)
        self.gripR.set_3d_properties(gripRz)
        
        self.gripL.set_data(gripLx, gripLy)
        self.gripL.set_3d_properties(gripLz)
        
        #all the points to check for collision
        return np.array([p2, p4, p5+g11, p5+g12, p5+g21, p5+g22])
    
    def animate(self, i):
        if(i < 100):
            position = np.array([ (i/2.5)  -20 ,25,10], dtype = np.float64)
        else:
            i = 100 - i
            position = np.array([ (i/2.5) + 20 ,25,10], dtype = np.float64)
            
        pose = pose3D(position, True)
        angles = inverseKinematics(pose)
        
        p1,p2,p4,p6 = forwardPosKinematics(angles)
        rot = forwardKinematicsRotation(angles)
        
        g00,g11,g12,g21,g22 = self.getGripper(rot)
        collisionPoints = self.setArm(p1,p2,p4,g00,g11,g12,g21,g22)
        
        self.arm.set_color('b')
        self.gripR.set_color('b')
        self.gripL.set_color('b')
        
        if(self.obstacles.size > 0):
            for box in self.obstacles:
                for i in range(0, collisionPoints.shape[0]):
                    box.checkColission(collisionPoints[i],self.radius) 
                    if( box.checkColission(collisionPoints[i],self.radius) == 0):
                        self.gripR.set_color('r')
                        self.gripL.set_color('r')
                        self.arm.set_color('r')
        
        return self.lines
    
    def runAnimation(self):
        return animation.FuncAnimation(self.fig, self.animate, 200, interval=25)

        