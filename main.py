from math import pi, sin, fabs
import kinematics as ki


if __name__ == "__main__":	

    x = sin(pi)
    if(fabs(x) < 0.001):
        x = 0;
        
    ki.test()