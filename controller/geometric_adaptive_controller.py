"""
Python implementation of Geometric Adaptive Controller based on:
Farhad A. Goodarzi, Daewon Lee, and Taeyoung Lee.
"Geometric Adaptive Tracking Control of a Quadrotor Unmanned Aerial Vehicle on SE(3) for Agile Maneuvers."
[DOI: 10.1115/1.4030419]
"""

import numpy as np
from scipy.spatial.transform import Rotation
from controller.controller_template import MultirotorControlTemplate
from controller.math import *
class GeometricAdaptiveController(MultirotorControlTemplate):
    def __init__(self, vehicle_params, dt=0.01):
        """
        Initialize the geometric adaptive controller.
        
        Parameters:
            vehicle_params: dict containing vehicle parameters
            dt: float, optional, default=0.01, time step for the controller
        """
        # Initialize parent class
        super().__init__(vehicle_params)
        
        # Controller gains
        self.kp = np.array([6.5,6.5,15])  # Position control gains
        self.kv = np.array([4.0, 4.0, 9])  # Velocity control gains
        self.kR = 544*np.ones(3)      # Attitude control gains
        self.kW = 46.64*np.ones(3)      # Angular velocity control gains
        self.kdW = np.array([20,20,40]) # Angular acceleration control gains
        self.katt_xy = 10                       # XY attitude gain
        self.katt_z = 2                        # Z attitude gain
        
        # Adaptive control parameters
        self.gamma_x = 2.0  # Adaptation gain for position
        self.gamma_R = 10.0  # Adaptation gain for attitude
        
        # Compute c1 and c2 based on system parameters and gains
        # Position adaptation parameter c1
        c1_option1 = np.sqrt(self.kp[0] / self.mass)
        c1_option2 = (4 * self.kp[0] * self.kv[0] / 
                     (self.kv[0] * self.kv[0] + 4 * self.mass * self.kp[0]))
        self.c1 = min(c1_option1, c1_option2)
        
        # Attitude adaptation parameter c2
        c2_option1 = np.sqrt(self.kR[0] / self.inertia[0,0]) / self.inertia[2,2]
        c2_option2 = (4 * self.kW[0] / 
                     (8 * self.kR[0] * self.inertia[2,2] + 
                      (self.kW[0] + self.inertia[0,0]) * (self.kW[0] + self.inertia[0,0])))
        self.c2 = min(c2_option1, c2_option2)

        
        self.B_theta_x = 10.0  # Position parameter bound
        self.W_x = np.eye(3)
        self.W_R = np.eye(3)
        
        # Initialize estimated uncertainties
        self.bar_theta_x = np.zeros(3)
        self.bar_theta_R = np.zeros(3)
        
        # Constants
        self.min_thrust = 0.1
        self.max_thrust = 20.0 

        # Time step for the controller
        self.dt = dt

    def update(self, t, state, flat_output):
        """
        Compute control inputs based on current state and desired flat outputs.
        
        Returns:
            dict containing 'cmd_thrust' and 'cmd_moment'
        """
      
        # Extract state and desired state
        R = Rotation.from_quat(state['q']).as_matrix()
        w = state['w']
        yaw_des = flat_output['yaw']
        
        ## Geometric Adaptive Position control    
        pos_err = flat_output['x'] - state['x']
        vel_err = flat_output['x_dot'] - state['v']
        
        # Compute desired acceleration
        F_des = self.mass *(self.kp * pos_err + 
                             self.kv * vel_err +   
                             flat_output['x_ddot'] + 
                             np.array([0, 0, self.g]) - 
                             self.W_x @ self.bar_theta_x)
        
        # Update position adaptation parameter 
        norm_theta_x = np.linalg.norm(self.bar_theta_x)
        Wx_ev_c1ex = self.W_x.T @ (-vel_err + self.c1 * -pos_err)
        if norm_theta_x < self.B_theta_x or (norm_theta_x == self.B_theta_x and self.bar_theta_x @ Wx_ev_c1ex <= 0):
            bar_theta_x_dot = self.gamma_x * Wx_ev_c1ex
        else:
            I_theta = np.eye(3) -np.outer(self.bar_theta_x, self.bar_theta_x) / norm_theta_x**2
            bar_theta_x_dot = self.gamma_x * I_theta @ Wx_ev_c1ex
        
        self.bar_theta_x += bar_theta_x_dot * self.dt

        # Desired thrust is force projects onto b3 axis.
        b3 = R @ self.e3
        u1 = np.dot(F_des, b3)

        # Desired attitude
        z_body = normalize(F_des)
        x_world = np.array([np.cos(yaw_des), np.sin(yaw_des), 0])
        y_body = np.cross(z_body, x_world)
        y_body = normalize(y_body)
        x_body = np.cross(y_body, z_body)
        R_des = np.vstack((x_body, y_body, z_body)).T
        eR = 0.5 * vee( R.T @ R_des - R_des.T @ R )

    
        # Compute desired angular velocity
        w_des = self.tilt_prioritized_control(state['q'], Rotation.from_matrix(R_des).as_quat())
        eW = w_des-w  
        
        ## Geometric Adaptive Attitude control
        u2 = self.inertia @ (self.kR * eR + \
                 self.kW * eW - \
                 self.W_R @ self.bar_theta_R  + \
                 self.kdW * eW )+ \
                 np.cross(w, self.inertia @ w)
        
        # Update attitude adaptation parameter
        bar_theta_R_dot = self.gamma_R * self.W_R.T @ (-eW + self.c2 * eR)
        self.bar_theta_R += bar_theta_R_dot * self.dt
        

         # Only some of these are necessary depending on your desired control abstraction. 
        cmd_motor_speeds = np.zeros((4,))
        cmd_motor_thrusts = np.zeros((4,))
        cmd_thrust = u1
        cmd_moment = u2
        cmd_q = np.array([0,0,0,1])
        cmd_w = w_des
        cmd_v = np.zeros((3,))

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_motor_thrusts':cmd_motor_thrusts,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q,
                         'cmd_w':cmd_w,
                         'cmd_v':cmd_v}
        
        return control_input

    def tilt_prioritized_control(self, q, q_des):
        """
        Based on:
        Dario Brescianini and Raffaello D’Andrea, 
        “Tilt-prioritized quadrocopter attitude control.”
        
        Args:
            q: Current quaternion [i,j,k,w]
            q_des: Desired quaternion [i,j,k,w]
        Returns:
            rate_cmd: Angular velocity command
        """
        # Get quaternion error (q_e = q^(-1) * q_des)
        q_curr = Rotation.from_quat(q)
        q_d = Rotation.from_quat(q_des)
        q_e = (q_curr.inv() * q_d).as_quat()  # [x,y,z,w]
        
        # Extract components
        qe_w = q_e[3]  # w component is last in scipy
        qe_x = q_e[0]
        qe_y = q_e[1]
        qe_z = q_e[2]
        
        # Compute temporary vector
        tmp = np.array([
            qe_w * qe_x - qe_y * qe_z,
            qe_w * qe_y + qe_x * qe_z,
            qe_z
        ])
        
        # Flip sign if w component is negative
        if qe_w <= 0:
            tmp[2] *= -1.0
        
        # Compute rate command
        T_att = np.array([self.katt_xy, self.katt_xy, self.katt_z])
        rate_cmd = 2.0 / np.sqrt(qe_w * qe_w + qe_z * qe_z) * T_att * tmp
        
        return rate_cmd

