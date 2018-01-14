import numpy as np
from numpy import sin, cos, sqrt, pi
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
    def get_r_prime(self, r_point):
        c, s = np.cos(self.angle), np.sin(self.angle)
        R = np.array([[c, -s, 0], [s, c, 0],[0, 0, 1]])
        
        r_temp = np.zeros(3)
        for i in range(0,3):
            r_temp[i] = r_point[i] - self.pos[i]
        
        R = R.transpose()
                
        return np.asarray(R.dot(r_temp))
    
    def check_colission(self, r_point, radius=0.0):
        r_prime = self.get_r_prime(r_point)
        x = r_prime[0]
        y = r_prime[1]
        z = r_prime[2]
        L, W, H = self.size
        
        if x < 0:
            dx = np.minimum(x + radius, 0.0)
        elif x > L:
            dx = np.maximum((x - radius) - L, 0.0)
        else:
            dx = 0.0
        
        if y < 0:
            dy = np.minimum(y + radius, 0.0)
        elif y > W:
            dy = np.maximum((y - radius) - W, 0.0)
        else:
            dy = 0.0
            
        if z < 0:
            dz = np.minimum(z + radius, 0.0)
        elif z > H:
            dz = np.maximum((z - radius) - H, 0.0)
        else:
            dz = 0.0
            
        r = sqrt(dx*dx + dy*dy + dz*dz)
        if r > 0.0001:
            return r, np.array([dx/r,dy/r,dz/r])
        else:
            return 0,np.array([0,0,0])





def plot_world():
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

    obstacles = np.array([])
        
    cut_off = 5.0 
    
    def __init__(self,fig,ax, radius = 0.0):
        self.ax = ax
        self.fig = fig
        self.arm, =   self.ax.plot([], [], [], 'yo-', lw=2)
        self.gripL, = self.ax.plot([], [], [], 'yo-', lw=2)
        self.gripR, = self.ax.plot([], [], [], 'yo-', lw=2)
        self.lines = [self.arm, self.gripL, self.gripR]
        self.radius = radius
        
        pose = pose3D(np.array([20,25,10]), True)
        self.goal_points = self.get_goal_points(pose)
        
    def get_control_points(self,p2, p4, rot):
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
        
        p5 = p4 + g00
        
        return np.array([p2, p4, p5+g11, p5+g12, p5+g21, p5+g22])
    
    def get_goal_points(self, goal_pose):
        
        angles = inverseKinematics(goal_pose)
        p1,p2,p3,p4,p6 = forwardPosKinematics(angles)
        rot = forwardKinematicsRotation(angles)
        goal_points = self.get_control_points(p2,p4,rot)
        
        return goal_points    
    
    def set_obstacles(self, obstacles):
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
    
    def set_arm(self, p1, p2, p3, p4, g00, g11, g12, g21, g22):
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
        return np.array([p3, p4, p5+g11, p5+g12, p5+g21, p5+g22])
    
    def get_attr_vecs(self, current_points, goal_points):
        size = current_points.shape[0]
        attr_vecs = np.zeros((size,3))
        r = np.zeros(size)
        
        for i in range(0,size):
            attr_vecs[i] = goal_points[i] - current_points[i]
            r[i] = np.linalg.norm(attr_vecs[i]) #normalize them
            if r[i] > 0:
                attr_vecs[i] /= r[i]
            
        return np.linalg.norm(r)
    
    
    def draw_arm(self, pose):
        angles = inverseKinematics(pose)
        p1,p2,p3,p4,p6 = forwardPosKinematics(angles)
        rot = forwardKinematicsRotation(angles)
        g00,g11,g12,g21,g22 = self.getGripper(rot)
        
        self.set_arm(p1,p2,p3,p4,g00,g11,g12,g21,g22)
        
        collisionPoints = np.array([p3,p4,p6])
        
        self.gripR.set_color('b')
        self.gripL.set_color('b')
        self.arm.set_color('b')
        
        size = collisionPoints.shape[0]
        if(self.obstacles.size > 0):
            for i in range(0, size):
                for box in self.obstacles:
                    r, temp_vec = box.check_colission(collisionPoints[i],self.radius) 
                    if(r == 0):
                        self.gripR.set_color('r')
                        self.gripL.set_color('r')
                        self.arm.set_color('r')
        
    
    def animate(self, i):
        if(i < 100):
            position = np.array([ (i/2.5)  -20 ,25,10], dtype = np.float64)
        else:
            i = 100 - i
            position = np.array([ (i/2.5) + 20 ,25,10], dtype = np.float64)
            
        pose = pose3D(position, True)
        angles = inverseKinematics(pose)
        
        p1,p2,p3,p4,p6 = forwardPosKinematics(angles)
        rot = forwardKinematicsRotation(angles)
        g00,g11,g12,g21,g22 = self.getGripper(rot)
        
        self.set_arm(p1,p3,p2,p4,g00,g11,g12,g21,g22)
        
        collisionPoints = np.array([p3,p4,p6])
        
        self.arm.set_color('b')
        self.gripR.set_color('b')
        self.gripL.set_color('b')
        
        size = collisionPoints.shape[0]
        r = np.zeros(size)
        vec = np.zeros((size,3))
        
        if(self.obstacles.size > 0):
            for i in range(0, size):
                for box in self.obstacles:
                    r, temp_vec = box.check_colission(collisionPoints[i],self.radius) 
                    if r < self.cut_off and r > 0:
                        vec[i] += temp_vec/(r*r)
                    elif(r == 0):
                        self.gripR.set_color('r')
                        self.gripL.set_color('r')
                        self.arm.set_color('r')

        for i in range(0, size):
            r = np.linalg.norm(vec[i])
            if r > 0:
                vec[i] /= r
        
        return self.lines
    
    
    def runAnimation(self):
        return animation.FuncAnimation(self.fig, self.animate, 200, interval=25)
    
    
    def set_animation_angles(self, animation_angles, steps):
        self.animation_angles = animation_angles
        self.steps = steps
    
    def animate_angles(self, i):
        angles = self.animation_angles[i]
        
        p1,p2,p3,p4,p6 = forwardPosKinematics(angles)
        rot = forwardKinematicsRotation(angles)
        g00,g11,g12,g21,g22 = self.getGripper(rot)
        
        collisionPoints = self.set_arm(p1,p3,p2,p4,g00,g11,g12,g21,g22)
        
        self.arm.set_color('b')
        self.gripR.set_color('b')
        self.gripL.set_color('b')
        
        size = collisionPoints.shape[0]
        
        if(self.obstacles.size > 0):
            for i in range(0, size):
                for box in self.obstacles:
                    r, temp_vec = box.check_colission(collisionPoints[i],self.radius) 
                    if(r == 0):
                        self.gripR.set_color('r')
                        self.gripL.set_color('r')
                        self.arm.set_color('r')
        
        return self.lines


    def runAnimation_angles(self):
        return animation.FuncAnimation(self.fig, self.animate_angles, self.steps, interval=50)


#--------------------------------------------------------------------------------------------------------------------------------------------

class simulation:
    alpha = 0.005 #amount by which to update the angels in radians
    
    def __init__(self,initial_pose, goal_pose, obstacles, radius = 0.0, cut_off = 0.0):
        self.initial_pose = initial_pose
        self.goal_pose = goal_pose
        self.obstacles = obstacles
        self.radius = radius #radius of all the control points
        self. cut_off = cut_off
        
    def reset(self):
        self.steps_taken = 0
        self.c_a = inverseKinematics(self.initial_pose) #current angles
        p1,p2,p3,p4,p6 = forwardPosKinematics(self.c_a)
        self.initial_points = np.array([p3,p4,p6])
        
        angles =  inverseKinematics(self.goal_pose) 
        p1,p2,p3,p4,p6 = forwardPosKinematics(angles)
        self.goal_points = np.array([p3,p4,p6])
        
        attr_vecs, self.attr_r = self.get_attr_vecs(self.initial_points, self.goal_points)
        self.prev_dist = self.attr_r
        
        rep_vecs, rep_forces, attr_vecs, dist, collision = self.get_current_state(self.c_a) 
        observation = rep_vecs.ravel()
        observation = np.append(observation,rep_forces)
        observation = np.append(observation,attr_vecs.ravel())
        return observation
    
    def setup_animation(self, fig,ax):
        self.ax = ax
        self.fig = fig
        self.arm, = self.ax.plot([], [], [], 'yo-', lw=2)
        
    def set_arm(self, p1, p2, p4, p6):
        thisx = [0,p1[0], p2[0], p4[0], p6[0]]
        thisy = [0,p1[1], p2[1], p4[1], p6[1]]
        thisz = [0,p1[2], p2[2], p4[2], p6[2]]
        self.arm.set_data(thisx, thisy)
        self.arm.set_3d_properties(thisz)
        
    def draw_arm(self, pose):
        angles = inverseKinematics(pose)
        p1,p2,p3,p4,p6 = forwardPosKinematics(angles)
        self.set_arm(p1,p2,p4,p6)
        self.get_current_state(angles)
        
    def clamp(self, angle, min_angle, max_angle):
        return np.clip(angle, min_angle, max_angle)
    
    #input current_angles and delta_angles
    def update_angles(self, c_a, d_a):
        #we get 1's and 0's but we need 1's and -1's
        for i in range(0,d_a.size):
            if d_a[i] == 0:
                d_a[i] = -1

        c_a[1] = self.clamp(c_a[1] + self.alpha*d_a[0],0,pi)
        c_a[2] = self.clamp(c_a[2] + self.alpha*d_a[1],0,pi)
        c_a[3] = self.clamp(c_a[3] + self.alpha*d_a[2],-pi/2,pi/2)
        c_a[4] = self.clamp(c_a[4] + self.alpha*d_a[3],-pi,pi)
        c_a[5] = self.clamp(c_a[5] + self.alpha*d_a[4],-pi,pi)
#        c_a[6] = self.clamp(c_a[6] + self.alpha*d_a[5],-pi,pi)
        return c_a
        
    def get_attr_vecs(self, current_points, goal_points):
        size = current_points.shape[0]
        attr_vecs = np.zeros((size,3))
        r = np.zeros(size)
        
        for i in range(0,size):
            attr_vecs[i] = goal_points[i] - current_points[i]
            r[i] = np.linalg.norm(attr_vecs[i]) #normalize them
            if r[i] > 0:
                attr_vecs[i] /= r[i]
        return attr_vecs, np.linalg.norm(r)
    
    def get_current_state(self,angles):
        p1,p2,p3,p4,p6 = forwardPosKinematics(angles)
        control_points = np.array([p3,p4,p6])
        size = control_points.shape[0]
        rep_vecs = np.zeros((size,3))
        rep_forces = np.zeros(size)

        collision = False
        
        #check obstacle collision and calculate repulsive vectors
        if(self.obstacles.size > 0):
            for i in range(0, size):
                for box in self.obstacles:
                    r, temp_vec = box.check_colission(control_points[i],self.radius) 
                    if r < self.cut_off and r > 0: 
                        rep_vecs[i] += temp_vec/(r*r) 
                        rep_forces[i] = self.clamp(1/(r*r),0,1)
                    elif r == 0:
                        collision = True
                        
        #limit the arm to it's workspace by checking if p6 is within the workspace sphere
        max_range = 47.0
        max_index = size - 1
        r = np.linalg.norm(control_points[max_index])
        
        if (max_range - (r + self.radius)) < self.cut_off:
            rep_vecs[max_index] -= control_points[max_index] #pull it back towards the origiin
            
        if  (r + self.radius) > max_range:
            collision = True
            
        #check collision with the ground
        for i in range(0, size):
            z = control_points[i][2]
            if z < self.cut_off and z > 0.001:
                force = self.clamp(1/(z*z),0,1)
                rep_vecs[i] += [0,0,1/(z*z)]
                rep_forces[i] = np.maximum(force, rep_forces[i])
            if z < 0:
                collision = True
        
        for i in range(0, size):
            r = np.linalg.norm(rep_vecs[i])
            if r > 0:
                rep_vecs[i] /= r
                
        attr_vecs, dist = self.get_attr_vecs(control_points, self.goal_points)        
        
        return rep_vecs, rep_forces, attr_vecs, dist, collision

    def step(self, action):
        update = self.bitarray(action,5)
        self.c_a = self.update_angles(self.c_a, update)
        reward = 0
        #get the observation based on the current angles
        rep_vecs, rep_forces, attr_vecs, dist, collision = self.get_current_state(self.c_a) 
       
        observation = rep_vecs.ravel()
        observation = np.append(observation,rep_forces)
        observation = np.append(observation,attr_vecs.ravel())
        
        #if collision we stop immediately and punish for it
        if collision:
            return observation, -100, True
        
        #get the reward based on the current action
        delta_dist = self.prev_dist - dist
        if delta_dist > 0:
            reward = 4*delta_dist
        else:
            reward = delta_dist
            
        self.prev_dist = dist
        
        
        done = False
        
        self.steps_taken += 1
        
        if self.steps_taken > 400:
            done = True
        
        #close enough to target is good enough
        if dist < 5:
            factor = self.steps_taken/500.0 
            reward += 250.0/factor
            reward += 250
            done = True
    
        if reward > 0 and reward < 0.05:
            reward = -0.05
    
        return observation, reward, done
    
    def bitarray(self, n, base):
        temp = np.array([1 if digit=='1' else 0 for digit in bin(n)[2:]])
        res = np.zeros(base)
        res[0:temp.size] = temp
        return res
    
    def generate_animation_angles(self, model):
        self.c_a = inverseKinematics(self.initial_pose)
        animation_angles = self.c_a
        state_size = 21
        state = self.reset()
        done = False
        while not done:
            state = np.reshape(state, [1, state_size])
            act_values = model.predict(state)
            action = np.argmax(act_values[0])
            state, reward, done = self.step(action)
            animation_angles = np.append(animation_angles, self.c_a)       
            print( action)
        return np.reshape(animation_angles, (-1,7))
        