# Module for implementation of controllers.
# 2016-12-12
import numpy as np

class PID(object):
    def __init__(self,Kp,Ki,Kd):
        self.Kp,self.Ki,self.Kd=Kp,Ki,Kd

    def der(self):
        return

    def cum_err(self):
        return
    
    def simulate(self,X,U0=0):
        """
        Simple PID controller implementation.
        2016-12-12
        
        X
        Signal
        """
        err=np.zeros_like(X)
        U=np.zeros_like(X)
        U[0]=U0
        err[0]=X[0]-U[0]
        
        for t in range(1,len(X)-1):
            err[t]=X[t]-U[t]
            der=(err[t]-err[t-1])
            U[t+1]=U[t]+(self.Kp*err[t] + self.Ki*err[:t].sum() + self.Kd*der)
            
        return U
    
    def _simulate(self,X,U0=0):
        """
        Simple PID controller implementation.
        2016-12-12
        
        X
        Signal
        """
        err=np.zeros_like(X)
        U=np.zeros_like(X)
        U[0]=U0
        err[0]=X[0]-U[0]
        delay=10
        
        err[:delay]=X[:delay]-U[:delay]
        for t in range(delay+1,len(X)-1):
            err[t]=X[t-delay]-U[t-delay]
            der=(err[t-delay]-err[t-delay-1])
            U[t+1]=U[t]+(self.Kp*err[t] + self.Ki*err[:t].sum() + self.Kd*der)
            
        return U
