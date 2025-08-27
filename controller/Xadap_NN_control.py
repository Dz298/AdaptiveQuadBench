"""
Python implementation of Learning-based Controller for Extreme Adaptation based on:
Dingqi Zhang, Antonio Loquercio, Jerry Tang, Ting-Hao Wang, Jitendra Malik, Mark W. Mueller.
"A Learning-based Quadcopter Controller with Extreme Adaptation"
IEEE Transactions on Robotics, 41(2):3948-3964, 2025.
https://doi.org/10.1109/TRO.2025.3577037
"""

# import numpy as np
from scipy.spatial.transform import Rotation
from controller.controller_template import MultirotorControlTemplate
from controller.geometric_control import GeoControl
from controller.quadrotor_control_mpc import ModelPredictiveControl
from rotorpy.controllers.quadrotor_control import SE3Control
from scipy.signal import butter
import numpy as np
import onnxruntime    # to inference ONNX models, we use the ONNX Runtime
from enum import Enum
import os
from collections import deque
import pandas as pd

class ModelType(Enum):
    BASE_MODEL = 1
    ADAP_MODULE = 2

class ButterworthFilter:
    """Low-pass filter for smoothing sensor measurements"""
    def __init__(self, cutoff_freq=30, sample_rate=100):
        self.fs = sample_rate
        self.fc = cutoff_freq
        self.b, self.a = butter(2, self.fc/(self.fs/2), btype='low')
        self.buffer = {i: deque([0.0]*10, maxlen=10) for i in range(3)}

    def run(self, x_new):
        filtered_x = np.zeros(3,)
        for i in range(3):
            self.buffer[i].append(x_new[i])
            y = (self.b[0] * self.buffer[i][-1] + 
                self.b[1] * self.buffer[i][-2] + 
                self.b[2] * self.buffer[i][-3] - 
                self.a[1] * self.buffer[i][-2] - 
                self.a[2] * self.buffer[i][-3]) / self.a[0]
            filtered_x[i] = y
        return filtered_x

class NeuralNetworkModel:
    """Neural network model for adaptation"""
    def __init__(self):
        # Load model paths
        current_path = os.path.dirname(os.path.abspath(__file__)) + '/models/Xadap/'
        self.base_model_path = current_path + 'mlp.onnx'
        self.adap_module_path = current_path + 'encoder.onnx'
        
        # Load normalization parameters
        self.obs_mean = np.loadtxt(current_path+'mean.csv', delimiter=" ")
        self.obs_var = np.loadtxt(current_path+'var.csv', delimiter=" ")
        
        # Action normalization parameters
        self.act_mean = np.array([0.5, 0.5, 0.5, 0.5])[np.newaxis, :]
        self.act_std = np.array([0.5, 0.5, 0.5, 0.5])[np.newaxis, :]
        
        # Initialize model parameters
        self.state_obs_size = 8
        self.act_size = 4
        self.history_len = 100
        
        # Initialize ONNX sessions
        self.base_session = None
        self.adap_session = None
        self.activate()

    def activate(self):
        """Initialize ONNX runtime sessions"""
        self.base_session = onnxruntime.InferenceSession(self.base_model_path, None)
        self.adap_session = onnxruntime.InferenceSession(self.adap_module_path, None)
        self.base_obs_name = self.base_session.get_inputs()[0].name
        self.adap_obs_name = self.adap_session.get_inputs()[0].name

    def normalize_obs(self, obs, model_type):
        """Normalize observations based on model type"""
        if model_type is ModelType.BASE_MODEL:
            return (obs - self.obs_mean[:self.state_obs_size+self.act_size]) / \
                   np.sqrt(self.obs_var[:self.state_obs_size+self.act_size] + 1e-8)
        else:
            obs_n = obs.reshape([1, -1])
            # Normalize state history
            state_history = obs_n[:, -self.history_len*(self.act_size+self.state_obs_size):-self.history_len*self.act_size]
            state_mean = np.tile(self.obs_mean[:self.state_obs_size], [1, self.history_len])
            state_var = np.tile(self.obs_var[:self.state_obs_size], [1, self.history_len])
            state_history_norm = (state_history - state_mean) / np.sqrt(state_var + 1e-8)
            obs_n[:, -self.history_len*(self.act_size+self.state_obs_size):-self.history_len*self.act_size] = state_history_norm
            
            # Normalize action history
            act_history = obs_n[:, -self.history_len*self.act_size:]
            act_mean = np.tile(self.obs_mean[self.state_obs_size:self.state_obs_size+self.act_size], [1, self.history_len])
            act_var = np.tile(self.obs_var[self.state_obs_size:self.state_obs_size+self.act_size], [1, self.history_len])
            act_history_norm = (act_history - act_mean) / np.sqrt(act_var + 1e-8)
            obs_n[:, -self.history_len*self.act_size:] = act_history_norm
            
            return obs_n

    def predict(self, cur_obs, last_act, obs_history, act_history):
        """Run inference through both networks"""
        # Normalize inputs
        norm_cur_obs = self.normalize_obs(np.concatenate((cur_obs, last_act)), ModelType.BASE_MODEL)
        norm_history = self.normalize_obs(np.concatenate((obs_history, act_history)), ModelType.ADAP_MODULE)
        
        # Get latent representation from adaptation module
        latent = self.adap_session.run(None, {self.adap_obs_name: norm_history})
        # Combine current observation with latent representation
        combined_input = np.concatenate((norm_cur_obs.reshape(1,-1), 
                                       np.asarray(latent).reshape((1,-1))), axis=1).astype(np.float32)
        
        # Get final prediction from base model
        raw_act = np.asarray(self.base_session.run(None, 
                            {self.base_obs_name: combined_input})).squeeze()
        
        # Denormalize output
        norm_action = (raw_act * self.act_std + self.act_mean)[0, :]
        return norm_action, raw_act

class Xadap_NN_control(MultirotorControlTemplate):
    """Main controller class combining high-level geometric control with neural network adaptation"""
    def __init__(self, vehicle_params):
        super().__init__(vehicle_params)
        
        # Initialize controllers and filters
        self.neural_network = NeuralNetworkModel()
        self.butterworth_filter = ButterworthFilter()
        
        # Control frequency parameters
        self.high_level_control_freq = 50.0
        self.next_t_high_level_control = 0.0

        # Initialize high-level geometric controller parameters, identical from the paper
        self._init_geometric_controller_params()
        # Initialize high-level controller from our control library
        self.high_level_control = ModelPredictiveControl(vehicle_params)
        
        # Initialize state histories
        self._init_state_histories()
        
        
        # Initialize command states
        self.cmd_w = np.zeros(3,)
        self.cmd_mass_norm_thrust = 0
        self.cmd_q = np.zeros(4,)
        self.cmd_moment = np.zeros(3,)

    def _init_geometric_controller_params(self):
        """Initialize geometric controller parameters"""
        # Position control parameters
        self.pos_control_nat_freq = 2.0
        self.pos_control_damping = 0.7
        self.pos_control_integral_gain = 1.0
        
        # Attitude control parameters
        self.att_control_timeconst_z = 1.0
        self.att_control_timeconst_xy = 0.05
        self.att_control_integral_gain = 0.0

        # Geo gains 
        self.kp_pos = np.array([5,5,30])
        self.kd_pos = np.array([4.0, 4.0, 9])
        self.kp_att = 544
        self.kd_att = 46.64
        
        # Control limits
        self._max_proper_acc = 20.0
        self._min_proper_acc = 0.1
        self._min_vertical_proper_acceleration = 0.5 * self.g
        
        # Integration parameters
        self._integral_dt = 1.0 / self.high_level_control_freq
        self.integral_pos_err = np.zeros(3)
        self.integral_yaw_err = 0.0

    def _init_state_histories(self):
        """Initialize state and action histories"""
        history_len = 100
        state_obs_size = 8
        act_size = 4
        
        self.cur_obs = np.zeros((state_obs_size,))
        self.last_act = np.zeros((act_size,))
        self.obs_history = deque([np.zeros(state_obs_size)]*history_len)
        self.act_history = deque([np.zeros(act_size)]*history_len)

    def update(self, t, state, flat_output):
        """Main control loop"""
        # Run high-level geometric controller at specified frequency
        if t > self.next_t_high_level_control:
            # # High-level control from our control library
            # high_level_control_input = self.high_level_control.update(t, state, flat_output)
            # self.cmd_mass_norm_thrust = high_level_control_input['cmd_thrust'] / self.mass

            # High-level control from paper
            high_level_control_input = self.embed_highlevel_run(
                state,flat_output
            )
            self.cmd_mass_norm_thrust = high_level_control_input['cmd_thrust']

            self.cmd_w = high_level_control_input['cmd_w']
            self.cmd_q = high_level_control_input['cmd_q']
            self.cmd_moment = high_level_control_input['cmd_moment']
            self.next_t_high_level_control = t + 1/self.high_level_control_freq

        # Process current state
        R = Rotation.from_quat(state['q']).as_matrix()
        meas_acc = self.butterworth_filter.run(state['accel'])
        
        # Prepare neural network input
        cur_obs = np.concatenate((
            state['w'],
            np.array([meas_acc[-1]], dtype=np.float32),
            self.cmd_w.reshape(-1,),
            np.array([self.cmd_mass_norm_thrust], dtype=np.float32)
        )).astype(np.float32)
        
        # Get motor commands from neural network
        norm_act, raw_act = self.neural_network.predict(
            cur_obs, self.last_act,
            np.asarray(self.obs_history, dtype=np.float32).flatten(),
            np.asarray(self.act_history, dtype=np.float32).flatten()
        )
        
        # Update histories
        self.obs_history.popleft()
        self.obs_history.append(cur_obs)
        self.act_history.popleft()
        self.act_history.append(raw_act)
        self.last_act = raw_act
        # Convert network output to motor commands
        spd_NN = norm_act.squeeze() * self.rotor_speed_max
        cmd_motor_speeds = self._reorder_motor_speeds(spd_NN)
        
        # Return control commands
        return {
            'cmd_motor_speeds': cmd_motor_speeds,
            'cmd_motor_thrusts': np.zeros(4,),
            'cmd_thrust': self.cmd_mass_norm_thrust,
            'cmd_w': self.cmd_w,
            'cmd_q': self.cmd_q,
            'cmd_moment': self.cmd_moment
        }

    def _reorder_motor_speeds(self, spd_NN):
        """Reorder motor speeds to match simulation model"""
        """ Drone model for NN:    

                  x
                  ^
             mot3+| mot0-
                  |
            y<----+-----
                  |
             mot1-| mot2+
           
            Drone model for Simulation:   

                  x
                  ^
             mot0+| mot1-
                  |
            y<----+-----
                  |
             mot3-| mot2+
        """
        spd_cmd = np.zeros(4,)
        spd_cmd[0] = spd_NN[3]  # mot0
        spd_cmd[1] = spd_NN[0]  # mot1
        spd_cmd[2] = spd_NN[2]  # mot2
        spd_cmd[3] = spd_NN[1]  # mot3
        return spd_cmd

    def update_trajectory(self, trajectory):
        """Update reference trajectory"""
        self.high_level_control.update_trajectory(trajectory)

    def embed_highlevel_run(self, state, flat_output):
        """
        High-level tilt-prioritized geometric controller that computes desired thrust and attitude.
        
        Args:
            state (dict): Current state
            flat_output (dict): Desired flat output
        
        Returns:
            dict: Control commands containing thrust and attitude
        """
        cur_pos = state['x']
        cur_vel = state['v']
        cur_att_vec = state['q']
        des_pos = flat_output['x']
        des_vel = flat_output['x_dot']
        des_acc = flat_output['x_ddot']
        des_yaw_angle = flat_output['yaw']
        des_yaw_rate = flat_output['yaw_dot']
        
        # Convert quaternion to rotation matrix
        cur_att = Rotation.from_quat(cur_att_vec).as_matrix()
        
        # Position control
        self.integral_pos_err += (des_pos - cur_pos) * self._integral_dt
        
        # Compute commanded acceleration
        cmd_acc = (des_pos - cur_pos) * self.kp_pos + \
                  (des_vel - cur_vel) * self.kd_pos + \
                  des_acc + self.integral_pos_err * self.pos_control_integral_gain
        # cmd_acc = (des_pos - cur_pos) * self.pos_control_nat_freq * self.pos_control_nat_freq + \
        #           (des_vel - cur_vel) * 2 * self.pos_control_nat_freq * self.pos_control_damping + \
        #           des_acc + self.integral_pos_err * self.pos_control_integral_gain


        cmd_proper_acc = cmd_acc + np.array([0, 0, self.g])
        
        # Saturate thrust
        norm_cmd_proper_acc = np.linalg.norm(cmd_proper_acc)
        if norm_cmd_proper_acc > self._max_proper_acc:
            cmd_proper_acc *= self._max_proper_acc / norm_cmd_proper_acc
            
        # Apply max tilt angle
        if cmd_proper_acc[2] < self._min_vertical_proper_acceleration:
            cmd_proper_acc[2] = self._min_vertical_proper_acceleration
            
        # Compute thrust direction
        cmd_thrust_dir = cmd_proper_acc / np.linalg.norm(cmd_proper_acc)
        
        # Compute total thrust
        out_cmd_thrust = np.linalg.norm(cmd_proper_acc) * np.dot(cur_att @ self.e3, cmd_thrust_dir)
        
        if out_cmd_thrust < self._min_proper_acc:
            out_cmd_thrust = self._min_proper_acc
            
        # Construct desired attitude
        cos_angle = np.dot(cmd_thrust_dir, self.e3)
        if cos_angle >= (1 - 1e-12):
            angle = 0
        elif cos_angle <= -(1 - 1e-12):
            angle = np.pi
        else:
            angle = np.arccos(cos_angle)
            
        rot_ax = np.cross(self.e3, cmd_thrust_dir)
        n = np.linalg.norm(rot_ax)
        
        if n < 1e-6:
            cmd_att = Rotation.from_matrix(np.eye(3))
        else:
            rot_ax = rot_ax / n
            cmd_att = Rotation.from_rotvec(angle * rot_ax)
        
        # Apply yaw rotation
        cmd_att_yawed = cmd_att * Rotation.from_rotvec([0, 0, des_yaw_angle])
        
        # Compute attitude error
        err_att = cmd_att_yawed.inv() * Rotation.from_quat(cur_att_vec)
        
        # Convert to axis-angle representation
        err_att_rotvec = err_att.as_rotvec()
        des_rot_vec = -err_att_rotvec  # Negative because we want the error from current to desired
        
        # Compute reduced attitude error
        des_red_att_rot_ax = np.cross(err_att.inv().as_matrix() @ self.e3, self.e3)
        des_red_att_rot_an_cos = np.dot(err_att.inv().as_matrix() @ self.e3, self.e3)
        
        if des_red_att_rot_an_cos >= 1.0:
            des_red_att_rot_an = 0
        elif des_red_att_rot_an_cos <= -1.0:
            des_red_att_rot_an = np.pi
        else:
            des_red_att_rot_an = np.arccos(des_red_att_rot_an_cos)
        
        # Normalize rotation axis
        n = np.linalg.norm(des_red_att_rot_ax)
        if n < 1e-12:
            des_red_att_rot_ax = np.zeros(3)
        else:
            des_red_att_rot_ax = des_red_att_rot_ax / n
        
        # Compute desired angular velocity
        k3 = 1.0 / self.att_control_timeconst_z
        k12 = 1.0 / self.att_control_timeconst_xy
        
        out_cmd_ang_vel = -k3 * des_rot_vec - (k12 - k3) * des_red_att_rot_an * des_red_att_rot_ax
        # Add integral term for yaw
        euler_error = (cmd_att_yawed.inv() * Rotation.from_quat(cur_att_vec)).inv().as_euler('ZYX')
        self.integral_yaw_err += euler_error[0] * self._integral_dt
        out_cmd_ang_vel[2] = des_yaw_rate
        
        return {
            'cmd_thrust': out_cmd_thrust,
            'cmd_q': cmd_att_yawed.as_quat(), 
            'cmd_w': out_cmd_ang_vel,
            'cmd_moment': np.zeros(3,)
        }


