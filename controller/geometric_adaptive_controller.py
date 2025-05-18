"""
Python implementation of Geometric Adaptive Controller based on:
Farhad A. Goodarzi, Daewon Lee, and Taeyoung Lee.
"Geometric Adaptive Tracking Control of a Quadrotor Unmanned Aerial Vehicle on SE(3) for Agile Maneuvers."
[DOI: 10.1115/1.4030419]
"""

import numpy as np
from scipy.spatial.transform import Rotation
from controller.controller_template import MultirotorControlTemplate
from controller.geometric_control import GeoControl
from controller.quadrotor_util import *
class GeometricAdaptiveController(GeoControl):
    def __init__(self, vehicle_params, dt=0.01):
        """
        Initialize the geometric adaptive controller.
        
        Parameters:
            vehicle_params: dict containing vehicle parameters
            dt: float, optional, default=0.01, time step for the controller
        """
        # Initialize parent class
        super().__init__(vehicle_params)
        
        # Adaptive control parameters
        self.gamma_x = 1  # Adaptation gain for position
        self.gamma_R = 0.1  # Adaptation gain for attitude
        
        # Compute c1 and c2 based on system parameters and gains
        # Position adaptation parameter c1
        c1_option1 = np.sqrt(self.k['x'][0] / self.mass)
        c1_option2 = (4 * self.k['x'][0] * self.k['v'][0] / 
                     (self.k['v'][0] * self.k['v'][0] + 4 * self.mass * self.k['x'][0]))
        self.c1 = min(c1_option1, c1_option2)
        
        # Attitude adaptation parameter c2
        c2_option1 = np.sqrt(self.k['R'][0] / self.inertia[0,0]) / self.inertia[2,2]
        c2_option2 = (4 * self.k['W'][0] / 
                     (8 * self.k['R'][0] * self.inertia[2,2] + 
                      (self.k['W'][0] + self.inertia[0,0]) * (self.k['W'][0] + self.inertia[0,0])))
        self.c2 = min(c2_option1, c2_option2)

        
        self.B_theta_x = 2.0  # Position parameter bound
        self.W_x = np.eye(3)
        self.W_R = np.eye(3)
        
        # Initialize estimated uncertainties
        self.bar_theta_x = np.zeros((3,1))
        self.bar_theta_R = np.zeros((3,1))
        
        # Time step for the controller
        self.dt = dt

        self.e3 = np.array([[0],[0],[1]])

    def update(self, t, state, flat_output):
        """
        Compute control inputs based on current state and desired flat outputs.
        """

        # Geometric control
        geo_control_input = super().update(t, state, flat_output)
        F_des = geo_control_input['cmd_thrust']
        moment_des = geo_control_input['cmd_moment'].reshape(3,1)

        error = geo_control_input['error']
        eW = error['W']
        eR = error['R']
        ex = error['x']
        ev = error['v']

        # Extract state and desired state
        R = Rotation.from_quat(state['q']).as_matrix()
        
        # Compute desired acceleration
        F_des = F_des + np.dot(np.ravel(self.W_x @ self.bar_theta_x),R@self.e3)
        # Update position adaptation parameter 
        norm_theta_x = np.linalg.norm(self.bar_theta_x)
        Wx_ev_c1ex = self.W_x.T @ (ev + self.c1 * ex)
        if norm_theta_x < self.B_theta_x or (norm_theta_x == self.B_theta_x and self.bar_theta_x @ Wx_ev_c1ex <= 0):
            bar_theta_x_dot = self.gamma_x * Wx_ev_c1ex
        else:
            I_theta = np.eye(3) -np.outer(self.bar_theta_x, self.bar_theta_x) / norm_theta_x**2
            bar_theta_x_dot = self.gamma_x * I_theta @ Wx_ev_c1ex
        
        self.bar_theta_x += bar_theta_x_dot * self.dt

        
        moment_des = moment_des - self.W_R @ self.bar_theta_R 
        
        # Update attitude adaptation parameter
        bar_theta_R_dot = self.gamma_R * self.W_R.T @ (eW + self.c2 * eR)
        self.bar_theta_R += bar_theta_R_dot * self.dt
        


        # Compute control input
        u = np.vstack((F_des, moment_des[0]))
        u = np.vstack((u, moment_des[1]))
        u = np.vstack((u, moment_des[2]))
        TM = np.array(u)
        cmd_rotor_thrusts = self.TM_to_f @ TM
        cmd_motor_speeds = cmd_rotor_thrusts / self.k_eta
        cmd_motor_speeds = np.sign(cmd_motor_speeds) * np.sqrt(np.abs(cmd_motor_speeds))
        cmd_thrust = F_des[0]
        cmd_moment = moment_des.reshape(3,)
        cmd_q = np.array([0,0,0,1])
        cmd_w = np.zeros((3,))
        cmd_v = np.zeros((3,))

        control_input = {'cmd_motor_speeds':cmd_motor_speeds.reshape(4,),
                         'cmd_motor_thrusts':cmd_rotor_thrusts.reshape(4,),
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q,
                         'cmd_w':cmd_w,
                         'cmd_v':cmd_v}
        
        return control_input

   
