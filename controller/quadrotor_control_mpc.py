"""
Python implementation of Model Predictive Controller based on:
Sihao Sun, Angel Romero, Philipp Foehn, Elia Kaufmann, and Davide Scaramuzza.
"A Comparative Study of Nonlinear MPC and Differential-Flatness-Based Control for Quadrotor Agile Flight"
[arXiv:2109.01365]
"""

import numpy as np
from time import time
from collections import deque
from scipy.spatial.transform import Rotation
from rotorpy.trajectories.hover_traj  import HoverTraj
from rotorpy.trajectories.circular_traj  import CircularTraj
from controller.quadrotor_mpc import QuadMPC
from controller.quadrotor_util import skew_symmetric, v_dot_q, quaternion_inverse
from controller.controller_template import MultirotorControlTemplate
class ModelPredictiveControl(MultirotorControlTemplate):
    """

    """
    def __init__(self, quad_params, trajectory=CircularTraj(radius=2), sim_rate=100, 
                 t_final=5, t_horizon=0.5, n_nodes=10 
                 ):
        """
        Parameters:
            quad_params, dict with keys specified in rotorpy/vehicles
        """
        super().__init__(quad_params)
        self.quad_params = quad_params
        self.quad_mpc = None
        
        self.t_final = t_final
        self.t_horizon = t_horizon
        self.n_nodes = n_nodes

        # compute optimation rate
        self.optimization_dt = t_horizon / n_nodes
        self.sim_dt = 1.0 / sim_rate
        self.sliding_index = 0 #determine current MPC reference

        # Initilize controls
        self.cmd_motor_forces = np.zeros((4,))

        # Time constant for angular velocity
        self.K_w_tau = np.linalg.inv(np.diag([20,20,0.8])) 
        # TODO: thread safety if we want to update MPC at runtime

    def update_trajectory(self, trajectory):
        self.quad_mpc = QuadMPC(quad_params=self.quad_params, trajectory=trajectory, t_final=self.t_final,
                                t_horizon=self.t_horizon, n_nodes=self.n_nodes)

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N 
                cmd_moment, N*m
                cmd_q, quaternion [i,j,k,w]
        """
        if self.quad_mpc is None:
            raise ValueError("QuadMPC is not initialized. Call update_trajectory() first.")
        # unpack state used for MPC
        state_mpc = self.unpack_state(state)

        task_index = None

        # Optimization loop
        index, _ = divmod(t, self.optimization_dt)
        if int(index) == self.sliding_index:
            self.quad_mpc.set_reference(self.sliding_index)
            w_opt,x_opt,sens_u = self.quad_mpc.run_optimization(initial_state=state_mpc, task_index=task_index)
            self.cmd_motor_forces = w_opt[:4]   # get controls
            cmd_motor_forces = self.cmd_motor_forces
            cmd_motor_speeds = cmd_motor_forces / self.k_eta
            cmd_motor_speeds = np.sign(cmd_motor_speeds) * np.sqrt(np.abs(cmd_motor_speeds))
            self.sliding_index += 1             # update slidng index
        # Compute motor speeds. Avoid taking square root of negative numbers.
        cmd_TM = self.f_to_TM @ self.cmd_motor_forces
        cmd_motor_forces = self.cmd_motor_forces
        cmd_motor_speeds = cmd_motor_forces / self.k_eta
        cmd_motor_speeds = np.sign(cmd_motor_speeds) * np.sqrt(np.abs(cmd_motor_speeds))
        cmd_thrust = cmd_TM[0]
        cmd_moment = np.array([cmd_TM[1], cmd_TM[2], cmd_TM[3]])
        cmd_w = self.K_w_tau @ (np.linalg.inv(self.inertia) @ (cmd_moment - np.cross(state['w'],self.inertia@state['w']))) + state['w']
        cmd_q = np.zeros((4,)) # 
        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_w':cmd_w,
                         'cmd_q':cmd_q}  # This dict is required by simulation env
        

        return control_input
    
    def unpack_state(self, state):
        """
        This function unpacks the state and returns an array [x, v, quaternion(wxyz), w] of shape (13,)
        """
        x = state['x']
        v = state['v']
        q_ = state['q']
        w = state['w']
        
        # Note: MPC uses quaternion as wxyz instead of xyzw used by the SIMULATOR
        q = np.array([q_[3], q_[0], q_[1], q_[2]])
        return np.concatenate([x,v,q,w])
    