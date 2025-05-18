"""
Python implementation of INDI Adaptive Controller based on:
Ewoud J. J. Smeur, Qiping Chu and Guido C. H. E. de Croon
"Adaptive incremental nonlinear dynamic inversion for attitude control of micro air vehicles"
[DOI:10.2514/1.G001490]

and

Ezra Tal and Sertac Karaman
"Accurate Tracking of Aggressive Quadrotor Trajectories Using Incremental Nonlinear Dynamic Inversion and Differential Flatness"
[DOI:10.1109/TCST.2020.3001117]
"""

import numpy as np
from scipy.spatial.transform import Rotation
from controller.controller_template import MultirotorControlTemplate
from controller.quadrotor_util import *
from rotorpy.controllers.quadrotor_control import SE3Control
from scipy.signal import butter
from collections import deque

class INDIAdaptiveController(MultirotorControlTemplate):
    def __init__(self, vehicle_params, dt=0.01):
        super().__init__(vehicle_params)
        self.dt = dt
        # Control Effectiveness Matrix
        self.G1 = np.ones((4,1))* 0.25 * self.mass / self.k_eta
        self.G2 = self.TM_to_f[:,1:] @ self.inertia / self.k_eta

        # Adaptive Gain
        self.mu = 1e-7

        # High-level Control
        self.high_level_control = SE3Control(vehicle_params)

        self.last_cmd_motor_speeds = np.zeros((4,1))
        self.last_meas_motor_speeds = np.zeros((4,1))
        self.last_meas_motor_speeds_dot = np.zeros((4,1))
        self.last_meas_omega = np.zeros((3,1))
        self.last_meas_alpha = np.zeros((3,1))

        self.delta_alpha = np.zeros((3,1))
        self.delta_thrust = np.zeros((1,1))

        # Butterworth filter parameters (30Hz cutoff, 100Hz sample rate)
        self.fs = 100  # Sample frequency (Hz)
        self.fc = 30   # Cut-off frequency (Hz)
        
        # Create second-order Butterworth filter coefficients
        self.b, self.a = butter(2, self.fc/(self.fs/2), btype='low')
        
        # Initialize filter states for motor speeds with larger buffer
        buffer_size = 10  
        self.motor_speed_buffer = {i: deque([0.0]*buffer_size, maxlen=buffer_size) for i in range(4)}
        self.filtered_motor_speeds = np.zeros((4,1))

    def butterworth_filter(self, x_new, buffer):
        """Apply second-order Butterworth filter"""
        buffer.append(x_new)
        y = (self.b[0] * buffer[-1] + 
             self.b[1] * buffer[-2] + 
             self.b[2] * buffer[-3] - 
             self.a[1] * buffer[-2] - 
             self.a[2] * buffer[-3]) / self.a[0]
        return y

    def update(self, t, state, flat_output):
        # Filter motor speeds
        meas_motor_speeds = state['rotor_speeds'].reshape(-1)
        for i in range(4):
            self.filtered_motor_speeds[i,0] = self.butterworth_filter(
                meas_motor_speeds[i], 
                self.motor_speed_buffer[i]
            )

        # Get high level control input
        high_level_control_input = self.high_level_control.update(t, state, flat_output)
        cmd_moment = high_level_control_input['cmd_moment'].reshape(-1,1)
        cmd_alpha = np.linalg.inv(self.inertia) @ (cmd_moment - np.cross(state['w'], 
                                                                        (self.inertia @ state['w'])).reshape(-1,1))
        cmd_mass_norm_thrust = high_level_control_input['cmd_thrust'] / self.mass

         # Estimated alpha 
        meas_alpha = ((state['w'].reshape(-1,1) - self.last_meas_omega) / self.dt)
        # Estimated acceleration in body frame
        meas_acc = state['accel']
        # Calculate motor commands
        mu = self.G1 * (cmd_mass_norm_thrust - np.linalg.norm(meas_acc)) + self.G2 @ (cmd_alpha - meas_alpha)
        cmd_motor_speeds = (self.filtered_motor_speeds ** 2 + mu)
        cmd_motor_speeds = np.sign(cmd_motor_speeds) * np.sqrt(np.abs(cmd_motor_speeds))

        cmd_motor_thrusts = self.k_eta * cmd_motor_speeds**2

        # Adaptive Law
        G = np.hstack([self.G1, self.G2])
        delta_state = np.vstack((self.delta_thrust, self.delta_alpha))
        G = G - self.mu * (G @ delta_state - (self.filtered_motor_speeds**2 - self.last_meas_motor_speeds**2)) @ delta_state.T

        self.G1 = G[:, :1]
        self.G2 = G[:, 1:]

        # Store states for next iteration
        self.last_cmd_motor_speeds = cmd_motor_speeds
        self.last_meas_motor_speeds = self.filtered_motor_speeds
        self.last_meas_motor_speeds_dot = (self.filtered_motor_speeds - self.last_meas_motor_speeds) / self.dt
        self.last_meas_alpha = meas_alpha
        self.last_meas_omega = state['w'].reshape(-1,1)
        self.delta_alpha = cmd_alpha - meas_alpha
        self.delta_thrust = cmd_mass_norm_thrust - np.linalg.norm(meas_acc)

        # Return control inputs
        control_input = {
            'cmd_motor_speeds': cmd_motor_speeds.reshape(-1),
            'cmd_motor_thrusts': cmd_motor_thrusts.reshape(-1),
            'cmd_thrust': high_level_control_input['cmd_thrust'],
            'cmd_moment': high_level_control_input['cmd_moment'],
            'cmd_q': high_level_control_input['cmd_q'],
            'cmd_w': high_level_control_input['cmd_w'],
        }
        
        return control_input