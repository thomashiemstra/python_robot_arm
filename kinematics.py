import numpy as np
from numpy import sqrt
from numpy import arctan2, sin , cos, pi

d1 = 12.5
d6 = 12.0
a2 = 15.0
d4 = 19.2

def inverseKinematics(pose):
    x,y,z = pose.position
    t = pose.orientation
    flip = pose.flip
    
    xc = x - d6*t[0, 2]
    yc = y - d6*t[1, 2]
    zc = z - d6*t[2, 2]
    angles = np.zeros(7, dtype = np.float32)


    angles[1] = arctan2(yc,xc)
    
    D = ( xc**2 + y**2 + (zc-d1)**2 - a2**2 - d4**2)/(2*a2*d4)
    angles[3] = arctan2(-sqrt(1 - D**2),D )

    k1 = a2+d4*cos(angles[3])
    k2 = d4*sin(angles[3])
    angles[2] = arctan2( (zc-d1), sqrt(xc**2 + yc**2) ) - arctan2(k2,k1) 

    angles[3] += pi/2;

    q1=angles[1]; q2=angles[2]; q3=angles[3]; q23 = q2 + q3;
    r11=t[0, 0]; r12=t[0, 1]; r13=t[0, 2]; r21=t[1, 0]; r22=t[1, 1]; r23=t[1, 2]; r31=t[2, 0]; r32=t[2, 1]; r33=t[2, 2];
   
    ax = r13*cos(q1)*cos(q23) + r23*cos(q23)*sin(q1) + r33*sin(q23)
    ay = -r23*cos(q1) + r13*sin(q1)
    az = -r33*cos(q23) + r13*cos(q1)*sin(q23) + r23*sin(q1)*sin(q23)
    sz = -r32*cos(q23) + r12*cos(q1)*sin(q23) + r22*sin(q1)*sin(q23)
    nz = -r31*cos(q23) + r11*cos(q1)*sin(q23) + r21*sin(q1)*sin(q23)

    if (flip):
        angles[4] = arctan2(-ay,-ax)
        angles[5] = arctan2(-sqrt(ax*ax+ay*ay),az)
        angles[6] = arctan2(-sz,nz)
    else:
        angles[4] = arctan2(ay,ax)
        angles[5] = arctan2(sqrt(ax*ax+ay*ay),az)
        angles[6] = arctan2(sz,-nz)
        
    return angles

def forwardPosKinematics(angles):
    q1=angles[1]; q2=angles[2]; q3=angles[3]; q4=angles[4]; q5=angles[5];
    
    p1= p2= p3= p4=p5= p6 = np.zeros(3, dtype = np.float64)
        
    p1[0] = 0;
    p1[1] = 0;
    p1[2] = d1;
    

    p2[0]  = a2*cos(q1)*cos(q2);
    p2[1]  = a2*cos(q2)*sin(q1);
    p2[2]  = d1+a2*sin(q2);
        
    p3[0] = cos(q1)*(a2*cos(q2) + (d4/2.0)*sin(q2+q3));
    p3[1] = sin(q1)*(a2*cos(q2) + (d4/2.0)*sin(q2+q3));
    p3[2] = d1 - (d4/2.0)*cos(q2+q3)+a2*sin(q2);
    
    p4[0] =  cos(q1)*(a2*cos(q2) + d4*sin(q2+q3));
    p4[1] =  sin(q1)*(a2*cos(q2) + d4*sin(q2+q3));
    p4[2] =  d1 - d4*cos(q2+q3) + a2*sin(q2);
    
    p5[0] = (d6/2.0)*sin(q1)*sin(q4)*sin(q5) + cos(q1)*(a2*cos(q2) + (d4 + (d6/2.0)*cos(q5))*sin(q2 + q3) + (d6/2.0)*cos(q2 + q3)*cos(q4)*sin(q5));
    p5[1] = cos(q3)*(d4 + (d6/2.0)*cos(q5))*sin(q1)*sin(q2) - (d6/2.0)*(cos(q4)*sin(q1)*sin(q2)*sin(q3) + cos(q1)*sin(q4))*sin(q5) + cos(q2)*sin(q1)*(a2 + (d4 + (d6/2.0)*cos(q5))*sin(q3) + (d6/2.0)*cos(q3)*cos(q4)*sin(q5));
    p5[2] = d1 - cos(q2 + q3)*(d4 + (d6/2.0)*cos(q5)) + a2*sin(q2) + (d6/2.0)*cos(q4)*sin(q2 + q3)*sin(q5);
    
    p6[0] = d6*sin(q1)*sin(q4)*sin(q5) + cos(q1)*(a2*cos(q2) + (d4 + d6*cos(q5))*sin(q2 + q3) + d6*cos(q2 + q3)*cos(q4)*sin(q5));
    p6[1] = cos(q3)*(d4 + d6*cos(q5))*sin(q1)*sin(q2) - d6*(cos(q4)*sin(q1)*sin(q2)*sin(q3) + cos(q1)*sin(q4))*sin(q5) + cos(q2)*sin(q1)*(a2 + (d4 + d6*cos(q5))*sin(q3) + d6*cos(q3)*cos(q4)*sin(q5));
    p6[2] =  d1 - cos(q2 + q3)*(d4 + d6*cos(q5)) + a2*sin(q2) + d6*cos(q4)*sin(q2 + q3)*sin(q5);
    
    return [p1,p2,p3,p4,p5,p6]

def forwardRotKinematics(angles):
    sx = cos(q6)*(cos(q4)*sin(q1) - cos(q1)*cos(q2 + q3)*sin(q4)) - (cos(q5)*sin(q1)*sin(q4) + cos(q1)*(cos(q2 + q3)*cos(q4)*cos(q5) - sin(q2 + q3)*sin(q5)))*sin(q6);
    sy = cos(q1)*(-cos(q4)*cos(q6) + cos(q5)*sin(q4)*sin(q6)) - sin(q1)*(-sin(q2 + q3)*sin(q5)*sin(q6) + cos(q2 + q3)*(cos(q6)*sin(q4) + cos(q4)*sin(q5)*sin(q6)));
    sz = -cos(q6)*sin(q2 + q3)*sin(q4) - (cos(q4)*cos(q5)*sin(q2 + q3) + cos(q2 + q3)*sin(q5))*sin(q6);
    ax = sin(q1)*sin(q4)*sin(q5) + cos(q1)*(cos(q5)*sin(q2 + q3) + cos(q2 + q3)*cos(q4)*sin(q5));
    ay = cos(q5)*sin(q1)*sin(q2 + q3) + (cos(q2 + q3)*cos(q4)*sin(q1) - cos(q1)*sin(q4))*sin(q5);
    az = -cos(q2 + q3)*cos(q5) + cos(q4)*sin(q2 + q3)*sin(q5);
    
    
class pose3D:
    
    position = np.array(3, dtype = np.float64)
    orientation = np.eye(3, dtype = np.float64)
    flip = False
    
    def __init__(self, position, orientation, flip):
        self.position = position
        self.orientation = orientation
        self.flip = flip
    
    def eulerMatrix(self, alpha, beta, gamma):
        ca = cos(alpha); cb = cos(beta); cy = cos(gamma);
        sa = sin(alpha); sb = sin(beta); sy = sin(gamma);
        self.orientation[0, 0] = ca*sb*cy + sa*sy;
        self.orientation[1, 0] = sa*sb*cy - ca*sy;
        self.orientation[2, 0] = cb*cy;
        
        self.orientation[0, 1] = ca*cb;
        self.orientation[1, 1] = sa*cb;
        self.orientation[2, 1] = -sb;
        
        self.orientation[0, 2] = ca*sb*sy - sa*cy;
        self.orientation[1, 2] = sa*sb*sy + ca*cy;
        self.orientation[2, 2] = cb*sy;
       
    





    