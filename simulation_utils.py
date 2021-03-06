import numpy as np
from numpy import sin, cos, sqrt, pi
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from kinematics import pose3D, inverseKinematics, forwardPosKinematics, forwardKinematicsRotation, d6


class box:
    size = np.zeros(3)
    pos = np.zeros(3)
    colission = False

    def __init__(self, size, pos=[0.0, 0.0, 0.0], angle=0.0, color='b'):
        self.size = size
        self.pos = pos
        self.angle = angle
        self.color = color
        self.L, self.W, self.H = self.size

    def setColor(self, color):
        self.color = color

    def cuboid_data(self):
        # code taken from
        # https://stackoverflow.com/a/35978146/4124317 
        #        pos = self.pos
        pos = np.zeros(3)
        for i in range(0, 3):
            pos[i] = self.pos[i] + self.size[i] / 2

        # suppose axis direction: x: to left; y: to inside; z: to upper
        # get the (left, outside, bottom) point
        o = [a - b / 2 for a, b in zip(pos, self.size)]
        # get the length, width, and height
        L, W, h = self.size

        Ca = cos(self.angle)
        Sa = sin(self.angle)

        x = np.matrix([[o[0], o[0] + L * Ca, o[0] + (L * Ca - W * Sa), o[0] - W * Sa, o[0]],
                       # x coordinate of points in bottom surface
                       [o[0], o[0] + L * Ca, o[0] + (L * Ca - W * Sa), o[0] - W * Sa, o[0]],
                       # x coordinate of points in upper surface
                       [o[0], o[0] + L * Ca, o[0] + L * Ca, o[0], o[0]],  # x coordinate of points in outside surface
                       [o[0] - W * Sa, o[0] + (L * Ca - W * Sa), o[0] + (L * Ca - W * Sa), o[0] - W * Sa,
                        o[0] - W * Sa]])  # x coordinate of points in inside surface

        y = np.matrix([[o[1], o[1] + L * Sa, o[1] + (L * Sa + W * Ca), o[1] + W * Ca, o[1]],
                       # y coordinate of points in bottom surface
                       [o[1], o[1] + L * Sa, o[1] + (L * Sa + W * Ca), o[1] + W * Ca, o[1]],
                       # y coordinate of points in upper surface
                       [o[1], o[1] + L * Sa, o[1] + L * Sa, o[1], o[1]],  # y coordinate of points in outside surface
                       [o[1] + W * Ca, o[1] + (L * Sa + W * Ca), o[1] + (L * Sa + W * Ca), o[1] + W * Ca,
                        o[1] + W * Ca]])  # y coordinate of points in inside surface

        z = np.matrix([[o[2], o[2], o[2], o[2], o[2]],
                       [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],
                       [o[2], o[2], o[2] + h, o[2] + h, o[2]],
                       [o[2], o[2], o[2] + h, o[2] + h, o[2]]])

        return x, y, z

    def plot(self, ax):
        X, Y, Z = self.cuboid_data()
        ax.plot_surface(X, Y, Z, color=self.color, rstride=1, cstride=1, alpha=1)

    #    vector of the point expressed in the cube's coordinate frame
    def get_r_prime(self, r_point):
        c, s = np.cos(self.angle), np.sin(self.angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        r_temp = np.zeros(3)
        for i in range(0, 3):
            r_temp[i] = r_point[i] - self.pos[i]

        R = R.transpose()

        return np.asarray(R.dot(r_temp))

    def check_colission(self, r_point, radius=0.001):
        r_prime = self.get_r_prime(r_point)

        if self.L > r_prime[0] > 0:
            dx = 0.0
        else:
            dx = r_prime[0]
            
        if self.W > r_prime[1] > 0:
            dy = 0.0
        else:
            dy = r_prime[1]    
        
        if self.H > r_prime[2] > 0:
            dz = 0.0
        else:
            dz = r_prime[2]   
            
        r = np.linalg.norm(np.array([dx,dy,dz]))
        
        if r > radius:
            return r, np.array([dx / r, dy / r, dz / r])
        else:
            return 0, np.array([0, 0, 0])



def plot_world():
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlim(-20, 20)
    ax.set_ylim(0, 40)
    ax.set_zlim(0, 40)
    len = 10
    ax.plot([0, len], [0, 0], [0, 0], 'o-', lw=2, color='r')
    ax.plot([0, 0], [0, len], [0, 0], 'o-', lw=2, color='g')
    ax.plot([0, 0], [0, 0], [0, len], 'o-', lw=2, color='b')
    ax.set_xlabel('X ---->')
    ax.set_ylabel('Y ---->')
    ax.set_zlabel('Z ---->')
    return fig, ax


class animateArm:
    obstacles = np.array([])
    
    def __init__(self, fig, ax, radius=0.0):
        self.ax = ax
        self.fig = fig
        self.arm, = self.ax.plot([], [], [], 'yo-', lw=2)
        self.gripL, = self.ax.plot([], [], [], 'yo-', lw=2)
        self.gripR, = self.ax.plot([], [], [], 'yo-', lw=2)
        self.lines = [self.arm, self.gripL, self.gripR]
        self.radius = radius

        pose = pose3D(np.array([20, 25, 10]), True)
        self.goal_points = self.get_goal_points(pose)

    def get_control_points(self, p2, p4, rot):
        temp = np.array([0, 0, d6 / 2], dtype=np.float64)
        g00 = rot.dot(temp)
        temp = np.array([0, -2, 0], dtype=np.float64)
        g11 = rot.dot(temp)
        temp = np.array([0, -2, d6 / 2], dtype=np.float64)
        g12 = rot.dot(temp)
        temp = np.array([0, 2, 0], dtype=np.float64)
        g21 = rot.dot(temp)
        temp = np.array([0, 2, d6 / 2], dtype=np.float64)
        g22 = rot.dot(temp)

        p5 = p4 + g00

        return np.array([p2, p4, p5 + g11, p5 + g12, p5 + g21, p5 + g22])

    def get_goal_points(self, goal_pose):

        angles = inverseKinematics(goal_pose)
        p1, p2, p3, p4, p6 = forwardPosKinematics(angles)
        rot = forwardKinematicsRotation(angles)
        goal_points = self.get_control_points(p2, p4, rot)

        return goal_points

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

    def getGripper(self, rot):
        temp = np.array([0, 0, d6 / 2], dtype=np.float64)
        g00 = rot.dot(temp)
        temp = np.array([0, -2, 0], dtype=np.float64)
        g11 = rot.dot(temp)
        temp = np.array([0, -2, d6 / 2], dtype=np.float64)
        g12 = rot.dot(temp)
        temp = np.array([0, 2, 0], dtype=np.float64)
        g21 = rot.dot(temp)
        temp = np.array([0, 2, d6 / 2], dtype=np.float64)
        g22 = rot.dot(temp)

        return g00, g11, g12, g21, g22

    def set_arm(self, p1, p2, p3, p4, g00, g11, g12, g21, g22):
        p5 = p4 + g00

        thisx = [0, p1[0], p2[0], p4[0], p5[0]]
        thisy = [0, p1[1], p2[1], p4[1], p5[1]]
        thisz = [0, p1[2], p2[2], p4[2], p5[2]]

        gripRx = [p5[0], p5[0] + g11[0], p5[0] + g12[0]]
        gripRy = [p5[1], p5[1] + g11[1], p5[1] + g12[1]]
        gripRz = [p5[2], p5[2] + g11[2], p5[2] + g12[2]]

        gripLx = [p5[0], p5[0] + g21[0], p5[0] + g22[0]]
        gripLy = [p5[1], p5[1] + g21[1], p5[1] + g22[1]]
        gripLz = [p5[2], p5[2] + g21[2], p5[2] + g22[2]]

        self.arm.set_data(thisx, thisy)
        self.arm.set_3d_properties(thisz)

        self.gripR.set_data(gripRx, gripRy)
        self.gripR.set_3d_properties(gripRz)

        self.gripL.set_data(gripLx, gripLy)
        self.gripL.set_3d_properties(gripLz)

        # all the points to check for collision
        return np.array([p3, p4, p5 + g11, p5 + g12, p5 + g21, p5 + g22])

    def set_animation_angles(self, animation_angles, steps):
        self.animation_angles = animation_angles
        self.steps = steps

    def animate_angles(self, i):
        angles = self.animation_angles[i]

        p1, p2, p3, p4, p6 = forwardPosKinematics(angles)
        rot = forwardKinematicsRotation(angles)
        g00, g11, g12, g21, g22 = self.getGripper(rot)

        self.set_arm(p1, p3, p2, p4, g00, g11, g12, g21, g22)
        collisionPoints = np.array([p3, p4, p6])

        self.arm.set_color('y')
        self.gripR.set_color('y')
        self.gripL.set_color('y')

        size = collisionPoints.shape[0]
                       
        if self.obstacles.size > 0:
            for i in range(0, size):
                for BOX in self.obstacles:
                    r, temp_vec = BOX.check_colission(collisionPoints[i], self.radius)
                    if (r == 0):
                        self.gripR.set_color('r')
                        self.gripL.set_color('r')
                        self.arm.set_color('r')                
        
        return self.lines

    def runAnimation_angles(self):
        return animation.FuncAnimation(self.fig, self.animate_angles, self.steps, interval=50)


# --------------------------------------------------------------------------------------------------------------------------------------------

class simulation:
    alpha = 0.1  #maximum amount by which to update the angels in radians
    num_control_points = 3
            
    initial_points = np.zeros((num_control_points, 3), dtype=np.float32)
    goal_points = np.zeros((num_control_points, 3), dtype=np.float32)
    prev_dist_arr = np.zeros(num_control_points, dtype=np.float32)
    initial_dist = np.zeros(num_control_points, dtype=np.float32)
    obs_dist_arr = np.zeros(num_control_points, dtype=np.float32)
    c_a = np.zeros(7, dtype=np.float32)
    quarter_way = False
    halfway = False
    almost_there = False

    def __init__(self, initial_pose, goal_pose, obstacles, radius=0.0, cut_off=0.0):
        self.initial_pose = initial_pose
        self.goal_pose = goal_pose
        self.obstacles = obstacles
        self.radius = radius  # radius of all the control points
        self.cut_off = cut_off
        self.steps_taken = 0

    def reset(self):
        self.steps_taken = 0
        self.c_a = inverseKinematics(self.initial_pose)  # current angles
        p1, p2, p3, p4, p6 = forwardPosKinematics(self.c_a)
        self.initial_points = np.array([p3, p4, p6])
        self.prev_pos_norm = np.linalg.norm(self.initial_points)

        angles = inverseKinematics(self.goal_pose)
        p1, p2, p3, p4, p6 = forwardPosKinematics(angles)
        self.goal_points = np.array([p3, p4, p6])

        attr_vecs, self.initial_dist_arr = self.get_attr_vecs(self.initial_points, self.goal_points)
        self.prev_dist_arr = self.initial_dist_arr
        self.quarter_way_arr = [False, False, False]
        self.halfway_arr = [False, False, False]
        self.almost_there_arr = [False, False, False]
        self.goal_reached_arr = [False, False, False]
        self.reward_factor = 2

        rep_vecs, rep_forces, attr_vecs, dist_3D, dist_arr, collision = self.get_current_state(self.c_a)
        self.prev_collective_dist = np.linalg.norm(dist_arr)
        observation = rep_vecs.ravel()
#        observation = np.append(observation, rep_forces)
        observation = np.append(observation, attr_vecs.ravel())
        observation = np.append(observation, dist_3D.ravel())
                
        for i in range(0,3):
            self.obs_dist_arr[i] = (dist_arr[i] / 20.0)
        
        observation = np.append(observation, self.obs_dist_arr.ravel())
        
        return observation

    @staticmethod
    def random_action():
        return np.random.uniform(-1, 1, 5)

    @staticmethod
    def clamp(val, min_val, max_val):
        return np.clip(val, min_val, max_val)

    # input current_angles and delta_angles
    def update_angles(self, c_a, d_a):

        c_a[1] = self.clamp(c_a[1] + self.alpha * d_a[0], 0, pi)
        c_a[2] = self.clamp(c_a[2] + self.alpha * d_a[1], 0, pi)
        c_a[3] = self.clamp(c_a[3] + self.alpha * d_a[2], -pi / 2, pi / 2)
        c_a[4] = self.clamp(c_a[4] + self.alpha * d_a[3], -pi, pi)
        c_a[5] = self.clamp(c_a[5] + self.alpha * d_a[4], -pi, pi)
        return c_a

    @staticmethod
    def get_3d_dist(current_points, goal_points):
        size = current_points.shape[0]
        dist_3D = np.zeros((size, 3))
        for i in range(0, size):
            dist_3D[i] = np.absolute((goal_points[i] - current_points[i]) / 20.0)

        return dist_3D

    @staticmethod
    def get_attr_vecs(current_points, goal_points):
        size = current_points.shape[0]
        attr_vecs = np.zeros((size, 3))
        r = np.zeros(size)
        
        cut_off = 4.0
        #when the control get's too close we reduce it's attractive vector by F(x)=e^(ax^2) - 1 with F(cut_off) = 1 and F(0) = 0
        a = np.log(2)/(cut_off*cut_off)

        for i in range(0, size):
            attr_vecs[i] = goal_points[i] - current_points[i]
            r[i] = np.linalg.norm(attr_vecs[i])  
            if r[i] > 0:
                attr_vecs[i] /= r[i] # normalize them
            if r[i] < cut_off:
                attr_vecs[i] *= np.exp(a*r[i]*r[i]) - 1
                
        return attr_vecs, r

    def get_current_state(self, angles):
        p1, p2, p3, p4, p6 = forwardPosKinematics(angles)
        control_points = np.array([p3, p4, p6])
        
#        pos_norm = np.linalg.norm(control_points)
#        collective_distance_traveled = self.prev_pos_norm - pos_norm
#        
#        if collective_distance_traveled > 0.1:
#            reward = 1
#        else:
#            reward = -1
#        
#        self.prev_pos_norm = pos_norm
        
        size = control_points.shape[0]
        rep_vecs = np.zeros((size, 3))
        rep_forces = np.zeros(size)

        collision = False

        # check obstacle collision and calculate repulsive vectors
        if self.obstacles.size > 0:
            for i in range(0, size):
                for BOX in self.obstacles:
                    r, temp_vec = BOX.check_colission(control_points[i], self.radius)
                    if self.cut_off > r > 0:
                        rep_vecs[i] += temp_vec / (r * r)
                        rep_forces[i] = self.clamp(1 / (r * r), 0, 1)
                    elif r == 0:
                        collision = True
                        
        # limit the arm to it's workspace by checking if p6 is within the workspace sphere
        max_range = 47.0
        max_index = size - 1
        r = np.linalg.norm(control_points[max_index])

        if (max_range - (r + self.radius)) < self.cut_off:
            rep_vecs[max_index] -= control_points[max_index]  # pull it back towards the origiin

        if (r + self.radius) > max_range:
            collision = True

        # check collision with the ground
        for i in range(0, size):
            z = control_points[i][2]
            if self.cut_off > z > 0.001:
                force = self.clamp(1 / (z * z), 0, 1)
                rep_vecs[i] += [0, 0, 1 / (z * z)]
                rep_forces[i] = np.maximum(force, rep_forces[i])
            if z < 0:
                collision = True

        for i in range(0, size):
            r = np.linalg.norm(rep_vecs[i])
            if r > 0:
                rep_vecs[i] /= r

        dist_3D = self.get_3d_dist(control_points, self.goal_points)
        
        attr_vecs, dist_arr = self.get_attr_vecs(control_points, self.goal_points)

        return rep_vecs, rep_forces, attr_vecs, dist_3D, dist_arr, collision

    def step(self, action):      
        for i in range(5):
            action[i] = self.clamp(action[i], -1, 1)
        reward = 0.0
        self.c_a = self.update_angles(self.c_a, action)
        # get the observation based on the current angles
        rep_vecs, rep_forces, attr_vecs, dist_3D, dist_arr, collision = self.get_current_state(self.c_a)

        observation = rep_vecs.ravel()
#        observation = np.append(observation, rep_forces)
        observation = np.append(observation, attr_vecs.ravel())
        observation = np.append(observation, dist_3D.ravel())
        
        for i in range(0,3):
            self.obs_dist_arr[i] = (dist_arr[i] / 20.0)
        
        observation = np.append(observation, self.obs_dist_arr.ravel())

        collective_dist = np.linalg.norm(dist_arr)
        
        reward += self.reward_factor*(self.prev_collective_dist - collective_dist)
        
        self.prev_collective_dist = collective_dist
        
        
        # if collision we stop immediately and punish for it
        if collision:
            return observation, -500, True, {}

        done = False
        
        self.steps_taken += 1
        if self.steps_taken > 200:
            print("collective_dist= ", collective_dist)
            done = True
        
        # get the reward based on the current action
        # the distance to the goal is checked per control point
        delta_dist = np.zeros(self.num_control_points, dtype=np.float32)
                
        for i in range(0, self.num_control_points):
            delta_dist[i] = self.prev_dist_arr[i] - dist_arr[i]
    
            reward += self.reward_factor*delta_dist[i]
            
            if not self.quarter_way_arr[i]:
                if dist_arr[i] < 3*self.initial_dist_arr[i] / 4:
                    reward += 25
                    self.quarter_way_arr[i] = True
    
            if not self.halfway_arr[i]:
                if dist_arr[i] < self.initial_dist_arr[i] / 2:
                    reward += 25
                    self.halfway_arr[i] = True
    
            if not self.almost_there_arr[i]:
                if dist_arr[i] < self.initial_dist_arr[i] / 4:
                    reward += 25
                    self.almost_there_arr[i] = True
            
            # as long as a point is close to it's goal we get an extra reward
            if dist_arr[i] < 2:
                reward += 0.5
                                              
                    
        self.prev_dist_arr = dist_arr
            
        # close enough to target is good enough
        if collective_dist < 3:
            factor = self.steps_taken / 100.0
            reward += 300.0 / factor
            reward += 100
            done = True
            print("goal reached!!!!!!!!!!")

        return observation, reward, done, {}

    def get_action(self, qstate, model, state_size):
        predX = np.zeros(shape=(1, state_size))
        predX[0] = qstate
        # print("trying to predict reward at qs_a", predX[0])
        pred = model.predict(predX[0].reshape(1, predX.shape[1]))
        return pred[0]


    def generate_animation_angles(self, model):
        state_size = 30
        state = self.reset()
        animation_angles = self.c_a
        done = False
        score = 0
        while not done:
            action = self.get_action(state, model, state_size)
            state, reward, done, _ = self.step(action)
            score += reward
            animation_angles = np.append(animation_angles, self.c_a)
        return np.reshape(animation_angles, (-1, 7)), score
