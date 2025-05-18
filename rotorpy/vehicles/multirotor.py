import numpy as np
from numpy.linalg import inv, norm
import scipy.integrate
from scipy.spatial.transform import Rotation
from rotorpy.vehicles.hummingbird_params import quad_params

import time

"""
Multirotor models
"""

def quat_dot(quat, omega):
    """
    Parameters:
        quat, [i,j,k,w]
        omega, angular velocity of body in body axes

    Returns
        duat_dot, [i,j,k,w]

    """
    # Adapted from "Quaternions And Dynamics" by Basile Graf.
    (q0, q1, q2, q3) = (quat[0], quat[1], quat[2], quat[3])
    G = np.array([[ q3,  q2, -q1, -q0],
                  [-q2,  q3,  q0, -q1],
                  [ q1, -q0,  q3, -q2]])
    quat_dot = 0.5 * G.T @ omega
    # Augment to maintain unit quaternion.
    quat_err = np.sum(quat**2) - 1
    quat_err_grad = 2 * quat
    quat_dot = quat_dot - quat_err * quat_err_grad
    return quat_dot

class Multirotor(object):
    """
    Multirotor forward dynamics model. 

    states: [position, velocity, attitude, body rates, wind, rotor speeds]

    Parameters:
        quad_params: a dictionary containing relevant physical parameters for the multirotor. 
        initial_state: the initial state of the vehicle. 
        control_abstraction: the appropriate control abstraction that is used by the controller, options are...
                                'cmd_motor_speeds': the controller directly commands motor speeds. 
                                'cmd_motor_thrusts': the controller commands forces for each rotor.
                                'cmd_ctbr': the controller commands a collective thrsut and body rates. 
                                'cmd_ctbm': the controller commands a collective thrust and moments on the x/y/z body axes
                                'cmd_ctatt': the controller commands a collective thrust and attitude (as a quaternion).
                                'cmd_vel': the controller commands a velocity vector in the world frame. 
                                'cmd_acc': the controller commands a mass normalized thrust vector (acceleration) in the world frame.
        aero: boolean, determines whether or not aerodynamic drag forces are computed. 
    """
    def __init__(self, quad_params, initial_state = {'x': np.array([0,0,0]),
                                            'v': np.zeros(3,),
                                            'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
                                            'w': np.zeros(3,),
                                            'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
                                            'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])},
                       control_abstraction='cmd_motor_speeds',
                       aero = True,  
                ):
        """
        Initialize quadrotor physical parameters.
        """

        # Inertial parameters
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.Ixy             = quad_params['Ixy']  # kg*m^2
        self.Ixz             = quad_params['Ixz']  # kg*m^2
        self.Iyz             = quad_params['Iyz']  # kg*m^2
        self.arm_length      = quad_params['arm_length']  # meters

        if 'com' in quad_params:
            self.com = quad_params['com']
        else:
            self.com = np.array([0.0, 0.0, 0.0])

        # Payload parameters
        self.payload_mass = 0 # kg
        self.payload_position = np.array([0.0, 0.0, 0.0]) # m

        # Frame parameters
        self.c_Dx            = quad_params['c_Dx']  # drag coeff, N/(m/s)**2
        self.c_Dy            = quad_params['c_Dy']  # drag coeff, N/(m/s)**2
        self.c_Dz            = quad_params['c_Dz']  # drag coeff, N/(m/s)**2

        self.num_rotors      = quad_params['num_rotors']
        self.rotor_pos       = quad_params['rotor_pos']

        self.rotor_dir       = quad_params['rotor_directions']

        self.extract_geometry()

        # Rotor parameters    
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s

        self.k_eta           = quad_params['k_eta']     # thrust coeff, N/(rad/s)**2
        self.k_m             = quad_params['k_m']       # yaw moment coeff, Nm/(rad/s)**2
        self.k_d             = quad_params['k_d']       # rotor drag coeff, N/(m/s)
        self.k_z             = quad_params['k_z']       # induced inflow coeff N/(m/s)
        self.k_flap          = quad_params['k_flap']    # Flapping moment coefficient Nm/(m/s)

        # Motor parameters
        self.tau_m           = quad_params['tau_m']     # motor reponse time, seconds
        self.motor_noise     = quad_params['motor_noise_std'] # noise added to the actual motor speed, rad/s / sqrt(Hz)

        # Additional constants.
        self.inertia = np.array([[self.Ixx, self.Ixy, self.Ixz],
                                 [self.Ixy, self.Iyy, self.Iyz],
                                 [self.Ixz, self.Iyz, self.Izz]])
        self.rotor_drag_matrix = np.array([[self.k_d,   0,                 0],
                                           [0,          self.k_d,          0],
                                           [0,          0,          self.k_z]])
        self.drag_matrix = np.array([[self.c_Dx,    0,          0],
                                     [0,            self.c_Dy,  0],
                                     [0,            0,          self.c_Dz]])
        self.aero_model = 'rotorpy'
        
        # Check if cd1_x exists in quad_params, if not set all related values to 0
        if 'cd1_x' in quad_params and quad_params['cd1_x'] is not None:
            self.aero_model = 'other'
            self.cdz_h = quad_params.get('cdz_h', 0)
            self.cd1x = quad_params['cd1_x']
            self.cd1y = quad_params['cd1_y']
            self.cd1z = quad_params['cd1_z']
            self.drag_matrix = np.array([[self.cd1x,    0,          0],
                                         [0,            self.cd1y,  0],
                                         [0,            0,          self.cd1z]])
        else:
            self.cdz_h = 0
            self.cd1x = 0
            self.cd1y = 0
            self.cd1z = 0
        
        # rotor efficiency
        if 'rotor_efficiency' in quad_params:
            self.rotor_efficiency = quad_params['rotor_efficiency']
        else:
            self.rotor_efficiency = np.ones(self.num_rotors)

        self.g = 9.81 # m/s^2

        self.inv_inertia = inv(self.inertia)
        self.weight = np.array([0, 0, -self.mass*self.g])

        # Control allocation
        k = self.k_m/self.k_eta  # Ratio of torque to thrust coefficient. 

        # Below is an automated generation of the control allocator matrix. It assumes that all thrust vectors are aligned
        # with the z axis.
        self.f_to_TM = np.vstack((np.ones((1,self.num_rotors)),np.hstack([np.cross(self.rotor_pos[key],np.array([0,0,1])).reshape(-1,1)[0:2] for key in self.rotor_pos]), (k * self.rotor_dir).reshape(1,-1)))
        self.TM_to_f = np.linalg.inv(self.f_to_TM)

        # Set the initial state
        self.initial_state = initial_state

        self.control_abstraction = control_abstraction

        self.k_w = 1                # The body rate P gain        (for cmd_ctbr)
        self.k_v = 10               # The *world* velocity P gain (for cmd_vel)
        self.kp_att = 544           # The attitude P gain (for cmd_vel, cmd_acc, and cmd_ctatt)
        self.kd_att = 46.64         # The attitude D gain (for cmd_vel, cmd_acc, and cmd_ctatt)

        self.aero = aero

    def extract_geometry(self):
        """
        Extracts the geometry in self.rotors for efficient use later on in the computation of 
        wrenches acting on the rigid body.
        The rotor_geometry is an array of length (n,3), where n is the number of rotors. 
        Each row corresponds to the position vector of the rotor relative to the CoM. 
        """
        
        self.rotor_geometry = np.array([]).reshape(0,3)
        for rotor in self.rotor_pos:
            # Adjust rotor position relative to the COM
            r = self.rotor_pos[rotor] - self.com
            self.rotor_geometry = np.vstack([self.rotor_geometry, r])
        
        return

    def statedot(self, state, control, t_step):
        """
        Integrate dynamics forward from state given constant cmd_rotor_speeds for time t_step.
        """

        cmd_rotor_speeds = self.get_cmd_motor_speeds(state, control)

        # The true motor speeds can not fall below min and max speeds.
        cmd_rotor_speeds = np.clip(cmd_rotor_speeds, self.rotor_speed_min, self.rotor_speed_max) 

        # Form autonomous ODE for constant inputs and integrate one time step.
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, cmd_rotor_speeds,state['ext_force'],state['ext_torque'])
        s = Multirotor._pack_state(state)
        
        s_dot = s_dot_fn(0, s)
        v_dot = s_dot[3:6]
        w_dot = s_dot[10:13]

        state_dot = {'vdot': v_dot,'wdot': w_dot}
        return state_dot 


    def step(self, state, control, t_step):
        """
        Integrate dynamics forward from state given constant control for time t_step.
        """

        cmd_rotor_speeds = self.get_cmd_motor_speeds(state, control)

        # Compensate for rotor efficiency
        cmd_rotor_speeds = cmd_rotor_speeds

        # The true motor speeds can not fall below min and max speeds.
        cmd_rotor_speeds = np.clip(cmd_rotor_speeds, self.rotor_speed_min, self.rotor_speed_max) 

        # Form autonomous ODE for constant inputs and integrate one time step.
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, cmd_rotor_speeds,state['ext_force'],state['ext_torque'])
        s = Multirotor._pack_state(state)

        # Option 1 - RK45 integration
        sol = scipy.integrate.solve_ivp(s_dot_fn, (0, t_step), s, first_step=t_step)
        s = sol['y'][:,-1]
        # Option 2 - Euler integration
        # s = s + s_dot_fn(0, s) * t_step  # first argument doesn't matter. It's time invariant model

        state_unpacked = Multirotor._unpack_state(s)

        # Re-normalize unit quaternion.
        state_unpacked['q'] = state_unpacked['q'] / norm(state_unpacked['q'])

        # Add noise to the motor speed measurement
        state_unpacked['rotor_speeds'] += np.random.normal(scale=np.abs(self.motor_noise), size=(self.num_rotors,))
        state_unpacked['rotor_speeds'] = np.clip(state_unpacked['rotor_speeds'], self.rotor_speed_min, self.rotor_speed_max)
        state_unpacked['ext_force'] = state['ext_force']
        state_unpacked['ext_torque'] = state['ext_torque']

        return state_unpacked

    def _s_dot_fn(self, t, s, cmd_rotor_speeds, ext_force, ext_torque):
        """
        Compute derivative of state for quadrotor given fixed control inputs as
        an autonomous ODE.
        """

        state = Multirotor._unpack_state(s)

        rotor_speeds = state['rotor_speeds']
        inertial_velocity = state['v']
        wind_velocity = state['wind']
        R = Rotation.from_quat(state['q']).as_matrix()

        # Rotor speed derivative
        rotor_accel = (1/self.tau_m)*(cmd_rotor_speeds - rotor_speeds)

        # Position derivative.
        x_dot = state['v']

        # Orientation derivative.
        q_dot = quat_dot(state['q'], state['w'])

        # Compute airspeed vector in the body frame
        body_airspeed_vector = R.T@(inertial_velocity - wind_velocity)

        # Compute total wrench in the body frame based on the current rotor speeds and their location w.r.t. CoM
        (FtotB, MtotB) = self.compute_body_wrench(state['w'], rotor_speeds, body_airspeed_vector)

        # Rotate the force from the body frame to the inertial frame
        Ftot = R@FtotB

        # Velocity derivative.
        v_dot = (self.weight + Ftot + ext_force) / self.mass

        # Angular velocity derivative.
        w = state['w']
        w_hat = Multirotor.hat_map(w)
        w_dot = self.inv_inertia @ (MtotB + ext_torque - w_hat @ (self.inertia @ w))

        # NOTE: the wind dynamics are currently handled in the wind_profile object. 
        # The line below doesn't do anything, as the wind state is assigned elsewhere. 
        wind_dot = np.zeros(3,)

        # Pack into vector of derivatives.
        s_dot = np.zeros((16+self.num_rotors,))
        s_dot[0:3]   = x_dot
        s_dot[3:6]   = v_dot
        s_dot[6:10]  = q_dot
        s_dot[10:13] = w_dot
        s_dot[13:16] = wind_dot
        s_dot[16:]   = rotor_accel

        return s_dot

    def compute_body_wrench(self, body_rates, rotor_speeds, body_airspeed_vector):
        """
        Computes the wrench acting on the rigid body based on the rotor speeds for thrust and airspeed 
        for aerodynamic forces. 
        The airspeed is represented in the body frame.
        The net force Ftot is represented in the body frame. 
        The net moment Mtot is represented in the body frame. 
        """

        # Get the local airspeeds for each rotor
        local_airspeeds = body_airspeed_vector[:, np.newaxis] + Multirotor.hat_map(body_rates)@(self.rotor_geometry.T) 

        # Scale k_eta for each rotor based on rotor efficiency
        k_eta_scaled = self.k_eta * self.rotor_efficiency
        # Scale k_m for each rotor based on rotor efficiency
        k_m_scaled = self.k_m * self.rotor_efficiency

        # Compute the thrust of each rotor with scaled k_eta
        T = np.array([0, 0, 1])[:, np.newaxis] * (k_eta_scaled * rotor_speeds**2)
        
        # Add in aero wrenches (if applicable)
        if self.aero:
            if self.aero_model == 'rotorpy':
                # Parasitic drag force acting at the CoM
                D = -Multirotor._norm(body_airspeed_vector)*self.drag_matrix@body_airspeed_vector
            else:
                # Parasitic drag force acting at the CoM
                D = -self.drag_matrix@body_airspeed_vector
                D[-1] += self.cdz_h*(body_airspeed_vector[0]**2 + body_airspeed_vector[1]**2)
            # Rotor drag (aka H force) acting at each propeller hub - scale with rotor efficiency
            H = -(rotor_speeds)*(self.rotor_drag_matrix@local_airspeeds)
            # Pitching flapping moment acting at each propeller hub - scale with rotor efficiency
            M_flap = -self.k_flap*(rotor_speeds)*((Multirotor.hat_map(local_airspeeds.T).transpose(2, 0, 1))@np.array([0,0,1])).T
        else:
            D = np.zeros(3,)
            H = np.zeros((3,self.num_rotors))
            M_flap = np.zeros((3,self.num_rotors))

        # Compute the moments due to the rotor thrusts, rotor drag (if applicable), and rotor drag torques
        M_force = -np.einsum('ijk, ik->j', Multirotor.hat_map(self.rotor_geometry), T+H)
        # Use scaled k_m for yaw moment
        M_yaw = self.rotor_dir*(np.array([0, 0, 1])[:, np.newaxis] * (k_m_scaled * rotor_speeds**2))

        # Sum all elements to compute the total body wrench
        FtotB = np.sum(T + H, axis=1) + D
        MtotB = M_force + np.sum(M_yaw + M_flap, axis=1)

        return (FtotB, MtotB)

    def get_cmd_motor_speeds(self, state, control):
        """
        Computes the commanded motor speeds depending on the control abstraction.
        For higher level control abstractions, we have low-level controllers that will produce motor speeds based on the higher level commmand. 

        """

        if self.control_abstraction == 'cmd_motor_speeds':
            # The controller directly controls motor speeds, so command that. 
            return control['cmd_motor_speeds']

        elif self.control_abstraction == 'cmd_motor_thrusts':
            # The controller commands individual motor forces. 
            cmd_motor_speeds = control['cmd_motor_thrusts'] / self.k_eta                        # Convert to motor speeds from thrust coefficient. 
            return np.sign(cmd_motor_speeds) * np.sqrt(np.abs(cmd_motor_speeds))

        elif self.control_abstraction == 'cmd_ctbm':
            # The controller commands collective thrust and moment on each axis. 
            cmd_thrust = control['cmd_thrust']
            cmd_moment = control['cmd_moment']  

        elif self.control_abstraction == 'cmd_ctbr':
            # The controller commands collective thrust and body rates on each axis. 

            cmd_thrust = control['cmd_thrust']

            # First compute the error between the desired body rates and the actual body rates given by state. 
            w_err = state['w'] - control['cmd_w']

            # Computed commanded moment based on the attitude error and body rate error
            wdot_cmd = -self.k_w*w_err
            cmd_moment = self.inertia@wdot_cmd

            # Now proceed with the cmd_ctbm formulation.

        elif self.control_abstraction == 'cmd_vel':
            # The controller commands a velocity vector. 
            
            # Get the error in the current velocity. 
            v_err = state['v'] - control['cmd_v']

            # Get desired acceleration based on P control of velocity error. 
            a_cmd = -self.k_v*v_err

            # Get desired force from this acceleration. 
            F_des = self.mass*(a_cmd + np.array([0, 0, self.g]))

            R = Rotation.from_quat(state['q']).as_matrix()
            b3 = R @ np.array([0, 0, 1])
            cmd_thrust = np.dot(F_des, b3)

            # Follow rest of SE3 controller to compute cmd moment. 

            # Desired orientation to obtain force vector.
            b3_des = F_des/np.linalg.norm(F_des)
            c1_des = np.array([1, 0, 0])
            b2_des = np.cross(b3_des, c1_des)/np.linalg.norm(np.cross(b3_des, c1_des))
            b1_des = np.cross(b2_des, b3_des)
            R_des = np.stack([b1_des, b2_des, b3_des]).T

            # Orientation error.
            S_err = 0.5 * (R_des.T @ R - R.T @ R_des)
            att_err = np.array([-S_err[1,2], S_err[0,2], -S_err[0,1]])

            # Angular control; vector units of N*m.
            cmd_moment = self.inertia @ (-self.kp_att*att_err - self.kd_att*state['w']) + np.cross(state['w'], self.inertia@state['w'])

        elif self.control_abstraction == 'cmd_ctatt':
            # The controller commands the collective thrust and attitude.

            cmd_thrust = control['cmd_thrust']

            # Compute the shape error from the current attitude and the desired attitude. 
            R = Rotation.from_quat(state['q']).as_matrix()
            R_des = Rotation.from_quat(control['cmd_q']).as_matrix()

            S_err = 0.5 * (R_des.T @ R - R.T @ R_des)
            att_err = np.array([-S_err[1,2], S_err[0,2], -S_err[0,1]])

            # Compute command moment based on attitude error. 
            cmd_moment = self.inertia @ (-self.kp_att*att_err - self.kd_att*state['w']) + np.cross(state['w'], self.inertia@state['w'])
        
        elif self.control_abstraction == 'cmd_acc':
            # The controller commands an acceleration vector (or thrust vector). This is equivalent to F_des in the SE3 controller. 
            F_des = control['cmd_acc']*self.mass

            R = Rotation.from_quat(state['q']).as_matrix()
            b3 = R @ np.array([0, 0, 1])
            cmd_thrust = np.dot(F_des, b3)

            # Desired orientation to obtain force vector.
            b3_des = F_des/np.linalg.norm(F_des)
            c1_des = np.array([1, 0, 0])
            b2_des = np.cross(b3_des, c1_des)/np.linalg.norm(np.cross(b3_des, c1_des))
            b1_des = np.cross(b2_des, b3_des)
            R_des = np.stack([b1_des, b2_des, b3_des]).T

            # Orientation error.
            S_err = 0.5 * (R_des.T @ R - R.T @ R_des)
            att_err = np.array([-S_err[1,2], S_err[0,2], -S_err[0,1]])

            # Angular control; vector units of N*m.
            cmd_moment = self.inertia @ (-self.kp_att*att_err - self.kd_att*state['w']) + np.cross(state['w'], self.inertia@state['w'])
        else:
            raise ValueError("Invalid control abstraction selected. Options are: cmd_motor_speeds, cmd_motor_thrusts, cmd_ctbm, cmd_ctbr, cmd_ctatt, cmd_vel, cmd_acc")

        # Take the commanded thrust and body moments and convert them to motor speeds
        TM = np.concatenate(([cmd_thrust], cmd_moment))               # Concatenate thrust and moment into an array
        cmd_motor_forces = self.TM_to_f @ TM                                                # Convert to cmd_motor_forces from allocation matrix
        cmd_motor_speeds = cmd_motor_forces / self.k_eta                                    # Convert to motor speeds from thrust coefficient. 
        cmd_motor_speeds = np.sign(cmd_motor_speeds) * np.sqrt(np.abs(cmd_motor_speeds))

        return cmd_motor_speeds

    @classmethod
    def rotate_k(cls, q):
        """
        Rotate the unit vector k by quaternion q. This is the third column of
        the rotation matrix associated with a rotation by q.
        """
        return np.array([  2*(q[0]*q[2]+q[1]*q[3]),
                           2*(q[1]*q[2]-q[0]*q[3]),
                         1-2*(q[0]**2  +q[1]**2)    ])

    @classmethod
    def hat_map(cls, s):
        """
        Given vector s in R^3, return associate skew symmetric matrix S in R^3x3
        In the vectorized implementation, we assume that s is in the shape (N arrays, 3)
        """
        if len(s.shape) > 1:  # Vectorized implementation
            return np.array([[ np.zeros(s.shape[0]), -s[:,2],  s[:,1]],
                             [ s[:,2],     np.zeros(s.shape[0]), -s[:,0]],
                             [-s[:,1],  s[:,0],     np.zeros(s.shape[0])]])
        else:
            return np.array([[    0, -s[2],  s[1]],
                             [ s[2],     0, -s[0]],
                             [-s[1],  s[0],     0]])

    @classmethod
    def _pack_state(cls, state):
        """
        Convert a state dict to Quadrotor's private internal vector representation.
        """
        s = np.zeros((20,))   # FIXME: this shouldn't be hardcoded. Should vary with the number of rotors. 
        s[0:3]   = state['x']       # inertial position
        s[3:6]   = state['v']       # inertial velocity
        s[6:10]  = state['q']       # orientation
        s[10:13] = state['w']       # body rates
        s[13:16] = state['wind']    # wind vector
        s[16:]   = state['rotor_speeds']     # rotor speeds

        return s

    @classmethod
    def _norm(cls, v):
        """
        Given a vector v in R^3, return the 2 norm (length) of the vector
        """
        norm = (v[0]**2 + v[1]**2 + v[2]**2)**0.5
        return norm

    @classmethod
    def _unpack_state(cls, s):
        """
        Convert Quadrotor's private internal vector representation to a state dict.
        x = inertial position
        v = inertial velocity
        q = orientation
        w = body rates
        wind = wind vector
        rotor_speeds = rotor speeds
        """
        state = {'x':s[0:3], 'v':s[3:6], 'q':s[6:10], 'w':s[10:13], 'wind':s[13:16], 'rotor_speeds':s[16:]}
        return state

    def update_payload(self, payload_mass, payload_position, payload_inertia=None):
        """
        Updates the payload parameters.
        
        Parameters:
            payload_mass: Mass of the payload in kg
            payload_position: Position of the payload's COM in the body frame
            payload_inertia: Inertia tensor of the payload about its COM (optional)
        """

        # Update payload parameters
        self.payload_mass = payload_mass
        self.payload_position = payload_position
        self.payload_inertia = payload_inertia

    def attach_payload(self):
        """
        Updates the center of mass and inertia when a payload is attached.
        
        Parameters:
            payload_mass: Mass of the payload in kg
            payload_position: Position of the payload's COM in the body frame
            payload_inertia: Inertia tensor of the payload about its COM (optional)
        """
        if self.payload_mass == 0:
            return np.zeros(3)
        
        # Store original values
        original_mass = self.mass
        original_com = self.com.copy()
        original_inertia = self.inertia.copy()
        
        # Calculate new total mass
        total_mass = original_mass + self.payload_mass
        
        # Calculate new COM position using weighted average
        new_com = (original_mass * original_com + self.payload_mass * self.payload_position) / total_mass
        
        # Update mass and COM
        self.mass = total_mass
        self.com = new_com
        # Update inertia tensor
        
        # 1. Shift the original inertia tensor to the new COM
        r_original = original_com - new_com  # Vector from new COM to original COM
        r_squared = np.sum(r_original**2)
        r_outer = np.outer(r_original, r_original)
        
        # Apply parallel axis theorem to shift original inertia to new COM
        shifted_original_inertia = original_inertia - original_mass * (r_squared * np.eye(3) - r_outer)
        
        # 2. Add the payload's contribution to the inertia
        if self.payload_inertia is not None:
            # If payload inertia is provided, shift it to the new COM
            r_payload = self.payload_position - new_com  # Vector from new COM to payload COM
            r_squared = np.sum(r_payload**2)
            r_outer = np.outer(r_payload, r_payload)
            
            # Apply parallel axis theorem to shift payload inertia to new COM
            shifted_payload_inertia = self.payload_inertia + self.payload_mass * (r_squared * np.eye(3) - r_outer)
            
            # Add to get total inertia
            self.inertia = shifted_original_inertia + shifted_payload_inertia
        else:
            # If no payload inertia provided, assume point mass
            r_payload = self.payload_position - new_com
            r_squared = np.sum(r_payload**2)
            r_outer = np.outer(r_payload, r_payload)
            
            # Point mass inertia contribution
            payload_inertia_contribution = self.payload_mass * (r_squared * np.eye(3) - r_outer)
            
            # Add to get total inertia
            self.inertia = shifted_original_inertia + payload_inertia_contribution
        
        # Update inverse inertia
        self.inv_inertia = np.linalg.inv(self.inertia)
        
        # Re-extract geometry with updated COM
        self.extract_geometry()
        
        # Update weight vector
        self.weight = np.array([0, 0, -self.mass*self.g])
        
        # Update control allocation matrix if needed
        # This is necessary because rotor positions relative to COM have changed
        k = self.k_m/self.k_eta
        self.f_to_TM = np.vstack((np.ones((1,self.num_rotors)),
                                 np.hstack([np.cross(self.rotor_pos[key] - self.com,np.array([0,0,1])).reshape(-1,1)[0:2] 
                                 for key in self.rotor_pos]), 
                                 (k * self.rotor_dir).reshape(1,-1)))
        self.TM_to_f = np.linalg.inv(self.f_to_TM)
        
        return new_com - original_com  # Return the COM shift
    
    def detach_payload(self):
        """
        Detaches the payload from the vehicle and restores original properties.
        
        This method:
        1. Restores the original mass, COM, and inertia
        2. Updates all dependent properties (weight, control allocation, etc.)
        3. Resets payload parameters
        
        Returns:
            The COM shift vector (original_com - previous_com)
        """
        if self.payload_mass == 0:
            return np.zeros(3)  # No payload to detach
        
        # Store current COM for calculating shift
        previous_com = self.com.copy()
        
        # Store payload info for return value
        detached_payload_mass = self.payload_mass
        detached_payload_position = self.payload_position.copy()
        
        # Restore original mass
        original_mass = self.mass - self.payload_mass
        self.mass = original_mass
        
        # Recalculate COM without payload
        if original_mass > 0:
            # Calculate original COM
            self.com = (self.mass * self.com - self.payload_mass * self.payload_position) / original_mass
        
        # Recalculate inertia tensor
        # 1. Remove payload contribution from inertia
        if self.payload_inertia is not None:
            # If payload inertia was provided, remove its shifted contribution
            r_payload = self.payload_position - previous_com
            r_squared = np.sum(r_payload**2)
            r_outer = np.outer(r_payload, r_payload)
            
            # Remove shifted payload inertia
            shifted_payload_inertia = self.payload_inertia + self.payload_mass * (r_squared * np.eye(3) - r_outer)
            self.inertia = self.inertia - shifted_payload_inertia
        else:
            # If payload was a point mass, remove its contribution
            r_payload = self.payload_position - previous_com
            r_squared = np.sum(r_payload**2)
            r_outer = np.outer(r_payload, r_payload)
            
            # Remove point mass inertia contribution
            payload_inertia_contribution = self.payload_mass * (r_squared * np.eye(3) - r_outer)
            self.inertia = self.inertia - payload_inertia_contribution
        
        # 2. Shift the inertia tensor to the new COM
        r_shift = previous_com - self.com  # Vector from new COM to previous COM
        r_squared = np.sum(r_shift**2)
        r_outer = np.outer(r_shift, r_shift)
        
        # Apply parallel axis theorem to shift inertia to new COM
        self.inertia = self.inertia - self.mass * (r_squared * np.eye(3) - r_outer)
        
        # Update inverse inertia
        self.inv_inertia = np.linalg.inv(self.inertia)
        
        # Re-extract geometry with updated COM
        self.extract_geometry()
        
        # Update weight vector
        self.weight = np.array([0, 0, -self.mass*self.g])
        
        # Update control allocation matrix
        k = self.k_m/self.k_eta
        self.f_to_TM = np.vstack((np.ones((1,self.num_rotors)),
                                np.hstack([np.cross(self.rotor_pos[key] - self.com,np.array([0,0,1])).reshape(-1,1)[0:2] 
                                for key in self.rotor_pos]), 
                                (k * self.rotor_dir).reshape(1,-1)))
        self.TM_to_f = np.linalg.inv(self.f_to_TM)
        
        # Reset payload parameters
        self.payload_mass = 0
        self.payload_position = np.zeros(3)
        self.payload_inertia = None
        
        # Return the COM shift
        return self.com - previous_com