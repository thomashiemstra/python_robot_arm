import numpy as np
from numpy import sqrt
from math import atan2, sin, cos, pi

d1 = 12.5
d6 = 12.0
a2 = 15.0
d4 = 19.2

def inverseKinematics(x, y, z, t, flip):
    xc = x - d6*t[0, 2]
    yc = y - d6*t[1, 2]
    zc = z - d6*t[2, 2]
    angles = np.zeros(7, dtype = np.float32)


    angles[1] = atan2(yc,xc)
    
    D = ( xc**2 + y**2 + (zc-d1)**2 - a2**2 - d4**2)/(2*a2*d4)
    angles[3] = atan2(-sqrt(1 - D**2),D )

    k1 = a2+d4*cos(angles[3])
    k2 = d4*sin(angles[3])
    angles[2] = atan2( (zc-d1), sqrt(xc**2 + yc**2) ) - atan2(k2,k1) 

    angles[3] += pi/2;

    q1=angles[1]; q2=angles[2]; q3=angles[3]; q23 = q2 + q3;
    r11=t[0, 0]; r12=t[0, 1]; r13=t[0, 2]; r21=t[1, 0]; r22=t[1, 1]; r23=t[1, 2]; r31=t[2, 0]; r32=t[2, 1]; r33=t[2, 2];
   
    ax = r13*cos(q1)*cos(q23) + r23*cos(q23)*sin(q1) + r33*sin(q23)
    ay = -r23*cos(q1) + r13*sin(q1)
    az = -r33*cos(q23) + r13*cos(q1)*sin(q23) + r23*sin(q1)*sin(q23)
    sz = -r32*cos(q23) + r12*cos(q1)*sin(q23) + r22*sin(q1)*sin(q23)
    nz = -r31*cos(q23) + r11*cos(q1)*sin(q23) + r21*sin(q1)*sin(q23)

    if(flip):
        angles[4] = atan2(-ay,-ax)
        angles[5] = atan2(-sqrt(ax*ax+ay*ay),az)
        angles[6] = atan2(-sz,nz)
	
    else:
        angles[4] = atan2(ay,ax)
        angles[5] = atan2(sqrt(ax*ax+ay*ay),az)
        angles[6] = atan2(sz,-nz)
        
    return angles
