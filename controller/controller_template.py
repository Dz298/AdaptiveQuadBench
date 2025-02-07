"""
Imports
"""
import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation  # This is a useful library for working with attitude.

class MultirotorControlTemplate(ABC):
    """
    Abstract base class for multirotor controllers.
    The controller is implemented with two required abstract methods: __init__() and update(). 
    The __init__() is used to instantiate the controller, and this is where any model parameters or 
    controller gains should be set. 
    In update(), the current time, state, and desired flat outputs are passed into the controller at 
    each simulation step. The output of the controller depends on the control abstraction for Multirotor...
        if cmd_motor_speeds: the output dict should contain the key 'cmd_motor_speeds'
        if cmd_motor_thrusts: the output dict should contain the key 'cmd_rotor_thrusts'
        if cmd_vel: the output dict should contain the key 'cmd_v'
        if cmd_ctatt: the output dict should contain the keys 'cmd_thrust' and 'cmd_q'
        if cmd_ctbr: the output dict should contain the keys 'cmd_thrust' and 'cmd_w'
        if cmd_ctbm: the output dict should contain the keys 'cmd_thrust' and 'cmd_moment'
    """
    def __init__(self, vehicle_params):
        """
        Constructor to save vehicle parameters.
        
        Parameters:
            vehicle_params, dict with keys specified in a python file under /rotorpy/vehicles/
        """
        # Quadrotor physical parameters.
        # Inertial parameters
        self.mass = vehicle_params['mass']  # kg
        self.Ixx = vehicle_params['Ixx']   # kg*m^2
        self.Iyy = vehicle_params['Iyy']   # kg*m^2
        self.Izz = vehicle_params['Izz']   # kg*m^2
        self.Ixy = vehicle_params['Ixy']   # kg*m^2
        self.Ixz = vehicle_params['Ixz']   # kg*m^2
        self.Iyz = vehicle_params['Iyz']   # kg*m^2

        # Frame parameters
        self.c_Dx = vehicle_params['c_Dx']  # drag coeff, N/(m/s)**2
        self.c_Dy = vehicle_params['c_Dy']  # drag coeff, N/(m/s)**2
        self.c_Dz = vehicle_params['c_Dz']  # drag coeff, N/(m/s)**2

        self.num_rotors = vehicle_params['num_rotors']
        self.rotor_pos = vehicle_params['rotor_pos']
        self.rotor_dir = vehicle_params['rotor_directions']

        # Rotor parameters    
        self.rotor_speed_min = vehicle_params['rotor_speed_min']  # rad/s
        self.rotor_speed_max = vehicle_params['rotor_speed_max']  # rad/s

        self.k_eta = vehicle_params['k_eta']      # thrust coeff, N/(rad/s)**2
        self.k_m = vehicle_params['k_m']          # yaw moment coeff, Nm/(rad/s)**2
        self.k_d = vehicle_params['k_d']          # rotor drag coeff, N/(m/s)
        self.k_z = vehicle_params['k_z']          # induced inflow coeff N/(m/s)
        self.k_flap = vehicle_params['k_flap']    # Flapping moment coefficient Nm/(m/s)

        # Motor parameters
        self.tau_m = vehicle_params['tau_m']      # motor reponse time, seconds

        # Common constants
        self.inertia = np.array([[self.Ixx, self.Ixy, self.Ixz],
                                [self.Ixy, self.Iyy, self.Iyz],
                                [self.Ixz, self.Iyz, self.Izz]])  # kg*m^2
        self.g = 9.81  # m/s^2

        # Linear map from individual rotor forces to scalar thrust and vector
        # moment applied to the vehicle.
        k = self.k_m/self.k_eta  # Ratio of torque to thrust coefficient. 

        # Below is an automated generation of the control allocator matrix. It assumes that all thrust vectors are aligned
        # with the z axis.
        self.f_to_TM = np.vstack((np.ones((1,self.num_rotors)),
                                  np.hstack([np.cross(self.rotor_pos[key],np.array([0,0,1])).reshape(-1,1)[0:2] for key in self.rotor_pos]), 
                                 (k * self.rotor_dir).reshape(1,-1)))
        self.TM_to_f = np.linalg.inv(self.f_to_TM)

        # Body Axis
        self.e1 = np.array([1, 0, 0])
        self.e2 = np.array([0, 1, 0])
        self.e3 = np.array([0, 0, 1])

    @abstractmethod
    def update(self, t, state, flat_output):
        """
        Abstract method that must be implemented by child classes.
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, the commanded speed for each motor, rad/s, (cmd_motor_speeds)
                cmd_thrust, the collective thrust of all rotors, N, (cmd_ctatt, cmd_ctbr, cmd_ctbm)
                cmd_moment, the control moments on each boxy axis, N*m, (cmd_ctbm)
                cmd_q, desired attitude as a quaternion [i,j,k,w], , (cmd_ctatt)
                cmd_w, desired angular rates in body frame, rad/s, (cmd_ctbr)
                cmd_v, desired velocity vector in world frame, m/s (cmd_vel)
        """
        pass