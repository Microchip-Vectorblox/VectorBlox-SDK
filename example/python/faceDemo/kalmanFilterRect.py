import numpy as np

class kalmanFilter:
    def __init__(self, box):
        obs = self.boxToObservation(box)
        self.x = np.matrix([obs[0,0], obs[1,0], obs[2,0], obs[3,0], 0, 0, 0, 0]).transpose() # state estimate [w,y,width,dx,dy,dwidth]
        dt = 1
        xAcc = 2
        yAcc = 2
        wAcc = 1
        hAcc = 1
        xVar = 50
        yVar = 50
        wVar = 200
        hVar = 200
        dxVar = 10
        dyVar = 10
        dwVar = 5
        dhVar = 5
        self.F = np.matrix(np.eye(8)) # state-transition model
        for n in range(0,4):
            self.F[n,n+4] = dt
        self.Q = np.matrix(np.zeros((8,8))) # covariance of the process noise
        self.Q[0,0] = xAcc*pow(dt,4)/4
        self.Q[1,1] = yAcc*pow(dt,4)/4
        self.Q[2,2] = wAcc*pow(dt,4)/4
        self.Q[3,3] = hAcc*pow(dt,4)/4
        self.Q[0,4] = self.Q[4,0] = xAcc*pow(dt,3)/2
        self.Q[1,5] = self.Q[5,1] = yAcc*pow(dt,3)/2
        self.Q[2,6] = self.Q[6,2] = wAcc*pow(dt,3)/2
        self.Q[3,7] = self.Q[7,3] = hAcc*pow(dt,3)/2
        self.Q[4,4] = xAcc*pow(dt,2)
        self.Q[5,5] = yAcc*pow(dt,2)
        self.Q[6,6] = wAcc*pow(dt,2)
        self.Q[7,7] = hAcc*pow(dt,2)
        
        self.H = np.matrix([[1, 0, 0, 0, 0, 0, 0, 0], # observation model
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0]])
        self.R = np.matrix(np.diag([xVar, yVar, wVar, hVar])) # covariance of the observational noise
        self.P = np.matrix(np.diag([xVar,yVar,wVar,hVar,dxVar,dyVar,dwVar,dhVar])) # covariance matrix estimate
        
    def boxToObservation(self, box):
        x = .5*(box[0]+box[2])
        y = .5*(box[1]+box[3])
        w = box[2]-box[0]
        h = box[3]-box[1]
        obs = np.matrix([x,y,w,h]).transpose()
        return obs
    
    def update(self, box):
        # Predict
        x_apri = self.F*self.x                  # a priori state estimate
        P_apri = self.F*self.P*np.transpose(self.F) + self.Q # a priori covariance estimate
        
        # Update
        if len(box)==0:
            self.x = x_apri
            self.P = P_apri
        else:
            z = self.boxToObservation(box)
            y = z - self.H*x_apri                   # innovation residual
            S = self.H*P_apri*np.transpose(self.H) + self.R     # innovation covariance
            K = P_apri*np.transpose(self.H)*np.linalg.pinv(S)   # optimal Kalman gain
            self.x = x_apri + K*y                   # a posteriori state estimate
            self.P = (np.eye(8)-K*self.H)*P_apri    # a posteriori covariance estimate

    def innovationResidual(self, box):
        z = self.boxToObservation(box)
        x_apri = self.F*self.x                  # x_apri = F*x
        y = z - self.H*x_apri
        return y
    
    def box(self):
        x = self.x[0,0]
        y = self.x[1,0]
        w = self.x[2,0]
        h = self.x[3,0]
        box = [x-w/2, y-h/2, x+w/2, y+h/2]
        return box
