import numpy as np
from time import time
from collections import deque
from scipy.spatial.transform import Rotation
from rotorpy.trajectories.hover_traj  import HoverTraj
from controller.quadrotor_mpc import QuadMPC
from controller.quadrotor_util import skew_symmetric, v_dot_q, quaternion_inverse
class ModelPredictiveControl(object):
    """

    """
    def __init__(self, quad_params, sim_rate, 
                 trajectory, t_final, t_horizon, n_nodes 
                 ):
        """
        Parameters:
            quad_params, dict with keys specified in rotorpy/vehicles
        """
        self.quad_mpc = QuadMPC(quad_params=quad_params, trajectory=trajectory, t_final=t_final,
                                t_horizon=t_horizon, n_nodes=n_nodes)

        # compute optimation rate
        self.optimization_dt = t_horizon / n_nodes
        self.sim_dt = 1/sim_rate
        self.sliding_index = 0 #determine current MPC reference

        # Initilize controls
        self.cmd_motor_forces = np.zeros((4,))

        # Load quad params
        self.num_rotors      = quad_params['num_rotors']
        self.rotor_pos       = quad_params['rotor_pos']
        self.k_eta           = quad_params['k_eta']     # thrust coeff, N/(rad/s)**2
        self.k_m             = quad_params['k_m']       # yaw moment coeff, Nm/(rad/s)**2
        k = self.k_m/self.k_eta
        self.f_to_TM = np.vstack((np.ones((1,self.num_rotors)),np.hstack([np.cross(self.rotor_pos[key],np.array([0,0,1])).reshape(-1,1)[0:2] for key in self.rotor_pos]), np.array([k*(-1)**i for i in range(self.num_rotors)]).reshape(1,-1)))
        self.TM_to_f = np.linalg.inv(self.f_to_TM)
        
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
        # unpack state used for MPC
        state = self.unpack_state(state)

        task_index = None

        # Optimization loop
        index, _ = divmod(t, self.optimization_dt)
        if int(index) == self.sliding_index:
            self.quad_mpc.set_reference(self.sliding_index)
            w_opt,x_opt,sens_u = self.quad_mpc.run_optimization(initial_state=state, task_index=task_index)
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
        cmd_q = np.zeros((4,)) # 
        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
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
    