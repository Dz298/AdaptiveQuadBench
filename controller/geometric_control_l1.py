"""
Python implementation of L1 Adaptive Geometric Controller based on:
Zhuohuan Wu, Sheng Cheng, Pan Zhao, Aditya Gahlawat, Kasey A. Ackerman, Arun Lakshmanan, Chengyu Yang, Jiahao Yu, Naira Hovakimyan.
"L1Quad: L1 Adaptive Augmentation of Geometric Control for Agile Quadrotors with Performance Guarantees"
[arXiv:2302.07208]
"""

from __future__ import print_function, division, absolute_import
import numpy as np
from scipy.spatial.transform import Rotation
from controller.geometric_control import GeoControl
# import jax
# import jax.numpy as np
from scipy.integrate import odeint
from scipy.integrate import ode
import numpy.linalg as la
import math
import pdb
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import sys
from time import perf_counter

class L1_GeoControl(GeoControl):
    """
    implementing the original geometric control
    """
    def __init__(self, quad_params):
        """
        Parameters:
            quad_params, dict with keys specified in rotorpy/vehicles
        """
        super().__init__(quad_params)

        """ L1-related parameters """
        self.As_v = -1 # parameter for L1
        self.As_omega = -1 # parameter for L1
        self.dt_L1 = 1/100 # sample time for L1 AC, for simplicity, set the same as the simulation step size

        """ For large uncertainties ..."""
        self.ctoffq1Thrust = 500 # cutoff frequency for thrust channel LPF (rad/s)
        self.ctoffq1Moment = 500 # cutoff frequency for moment channels LPF1 (rad/s)
        self.ctoffq2Moment = 500 # cutoff frequency for moment channels LPF2 (rad/s)

        self.L1_params = (self.As_v, self.As_omega, self.dt_L1, self.ctoffq1Thrust, self.ctoffq1Moment, self.ctoffq2Moment, self.mass, self.g, self.inertia )

        """ Initialization of L1 inputs """
        self.v_hat_prev = np.array([0.0, 0.0, 0.0])
        self.omega_hat_prev = np.array([0.0, 0.0, 0.0])
        self.R_prev = np.eye(3) 
        self.v_prev = np.array([0.0,0.0,0.0])
        self.omega_prev = np.array([0.0,0.0,0.0])

        self.u_b_prev = np.array([0.0,0.0,0.0,0.0])
        self.u_ad_prev = np.array([0.0,0.0,0.0,0.0])
        self.sigma_m_hat_prev = np.array([0.0,0.0,0.0,0.0])
        self.sigma_um_hat_prev = np.array([0.0,0.0])
        self.lpf1_prev = np.array([0.0,0.0,0.0,0.0])
        self.lpf2_prev = np.array([0.0,0.0,0.0,0.0])
        self.din_L1 = (self.v_hat_prev, self.omega_hat_prev, self.R_prev, self.v_prev, self.omega_prev,
                       self.u_b_prev, self.u_ad_prev, self.sigma_m_hat_prev, self.sigma_um_hat_prev, 
                       self.lpf1_prev, self.lpf2_prev)

    
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
                cmd_motor_speeds, rad/s
                cmd_motor_thrusts, N
                cmd_thrust, N 
                cmd_moment, N*m
                cmd_q, quaternion [i,j,k,w]
                cmd_w, angular rates in the body frame, rad/s
                cmd_v, velocity in the world frame, m/s
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        def L1AC(self, R, W, x, v, f, M):
            (As_v, As_omega, dt, ctoffq1Thrust, ctoffq1Moment, ctoffq2Moment, kg_vehicleMass, GRAVITY_MAGNITUDE, J ) = self.L1_params
            (v_hat_prev, omega_hat_prev, R_prev, v_prev, omega_prev,
            u_b_prev, u_ad_prev, sigma_m_hat_prev, sigma_um_hat_prev,
            lpf1_prev, lpf2_prev) = self.din_L1

            # == begin L1 adaptive control ==
            # first do the state predictor
            e3 = np.array([0.0, 0.0, 1.0])
            # load translational velocity
            v_now = v

            # load rotational velocity
            omega_now = W

            massInverse = 1.0 / kg_vehicleMass

            # compute prediction error (on previous step)
            vpred_error_prev = v_hat_prev - v_prev # computes v_tilde for (k-1) step
            omegapred_error_prev = omega_hat_prev - omega_prev # computes omega_tilde for (k-1) step

            v_hat = v_hat_prev + (-e3 * GRAVITY_MAGNITUDE + R_prev[:,2]* (u_b_prev[0] + u_ad_prev[0] + sigma_m_hat_prev[0]) * massInverse + R_prev[:,0] * sigma_um_hat_prev[0] * massInverse + R_prev[:,1] * sigma_um_hat_prev[1] * massInverse + vpred_error_prev * As_v) * dt
           
            Jinv = la.inv(J)
            # temp vector: thrustMomentCmd[1--3] + u_ad_prev[1--3] + sigma_m_hat_prev[1--3]
            # original form
            tempVec = np.array([u_b_prev[1] + u_ad_prev[1] + sigma_m_hat_prev[1], u_b_prev[2] + u_ad_prev[2] + sigma_m_hat_prev[2], u_b_prev[3] + u_ad_prev[3] + sigma_m_hat_prev[3]])
            omega_hat = omega_hat_prev + (-np.matmul(Jinv, np.cross(omega_prev, (np.matmul(J, omega_prev)))) + np.matmul(Jinv, tempVec) + omegapred_error_prev * As_omega) * dt

            # update the state prediction storage
            v_hat_prev = v_hat
            omega_hat_prev = omega_hat

            # compute prediction error (for this step)
            vpred_error = v_hat - v_now
            omegapred_error = omega_hat - omega_now

            # exponential coefficients coefficient for As
            exp_As_v_dt = math.exp(As_v * dt)
            exp_As_omega_dt = math.exp(As_omega * dt)

            # latter part of uncertainty estimation (piecewise constant) (step2: adaptation law)
            PhiInvmu_v = vpred_error / (exp_As_v_dt - 1) * As_v * exp_As_v_dt
            PhiInvmu_omega = omegapred_error / (exp_As_omega_dt - 1) * As_omega * exp_As_omega_dt

            sigma_m_hat = np.array([0.0,0.0,0.0,0.0]) # estimated matched uncertainty
            sigma_m_hat_2to4 = np.array([0.0,0.0,0.0]) # second to fourth element of the estimated matched uncertainty
            sigma_um_hat = np.array([0.0,0.0]) # estimated unmatched uncertainty

            sigma_m_hat[0] = -np.dot(R[:,2], PhiInvmu_v) * kg_vehicleMass # don't forget the minus
            # turn np.dot(R[:,2], PhiInvmu_v) * kg_vehicleMass to -np.dot(R[:,2], PhiInvmu_v) * kg_vehicleMass
            sigma_m_hat_2to4 = -np.matmul(J, PhiInvmu_omega)
            sigma_m_hat[1] = sigma_m_hat_2to4[0]
            sigma_m_hat[2] = sigma_m_hat_2to4[1]
            sigma_m_hat[3] = sigma_m_hat_2to4[2]

            sigma_um_hat[0] = -np.dot(R[:,0], PhiInvmu_v) * kg_vehicleMass
            sigma_um_hat[1] = -np.dot(R[:,1], PhiInvmu_v) * kg_vehicleMass

            # store uncertainty estimations
            sigma_m_hat_prev = sigma_m_hat
            sigma_um_hat_prev = sigma_um_hat

            # compute lpf1 coefficients
            lpf1_coefficientThrust1 = math.exp(- ctoffq1Thrust * dt)
            lpf1_coefficientThrust2 = 1.0 - lpf1_coefficientThrust1

            lpf1_coefficientMoment1 = math.exp(- ctoffq1Moment * dt)
            lpf1_coefficientMoment2 = 1.0 - lpf1_coefficientMoment1

            # update the adaptive control
            u_ad_int = np.array([0.0,0.0,0.0,0.0])
            u_ad = np.array([0.0,0.0,0.0,0.0])

            # low-pass filter 1 (negation is added to u_ad_prev to filter the correct signal)
            u_ad_int[0] = lpf1_coefficientThrust1 * (lpf1_prev[0]) + lpf1_coefficientThrust2 * sigma_m_hat[0]
            u_ad_int[1:4] = lpf1_coefficientMoment1 * (lpf1_prev[1:4]) + lpf1_coefficientMoment2 * sigma_m_hat[1:4]

            lpf1_prev = u_ad_int # store the current state

            # coefficients for the second LPF on the moment channel
            lpf2_coefficientMoment1 = math.exp(- ctoffq2Moment * dt)
            lpf2_coefficientMoment2 = 1.0 - lpf2_coefficientMoment1

            # low-pass filter 2 (optional)
            u_ad[0] = u_ad_int[0] # only one filter on the thrust channel
            u_ad[1:4] = lpf2_coefficientMoment1 * lpf2_prev[1:4] + lpf2_coefficientMoment2 * u_ad_int[1:4]

            lpf2_prev = u_ad # store the current state

            u_ad = -u_ad

            # store the values for next iteration (negation is added to u_ad_prev to filter the correct signal)
            u_ad_prev = u_ad

            v_prev = v_now
            omega_prev = omega_now
            R_prev = R
            u_b_prev = np.array([f,M[0],M[1],M[2]])

            controlcmd_L1 = np.array([f,M[0],M[1],M[2]]) + u_ad_prev

            self.din_L1 = (v_hat_prev, omega_hat_prev, R_prev, v_prev, omega_prev,
            u_b_prev, u_ad_prev, sigma_m_hat_prev, sigma_um_hat_prev,
            lpf1_prev, lpf2_prev)

            f_L1 = controlcmd_L1[0]
            M_L1 = controlcmd_L1[1:4]
            return (f_L1, M_L1, sigma_m_hat)
        baseline_control_input = super().update(t, state, flat_output)
        f = baseline_control_input['cmd_thrust']
        M = baseline_control_input['cmd_moment']

        x = state['x'].reshape(3)
        v = state['v'].reshape(3)
        R = Rotation.from_quat(state['q']).as_matrix()     
        W = state['w'].reshape(3)

        f_l1, M_l1, sigma_m_hat = L1AC(self,R,W,x,v,f,M)

        u_new = np.vstack((f_l1.reshape(1,1),M_l1.reshape(3,1)))

        TM = np.array(u_new)
        cmd_rotor_thrusts = self.TM_to_f @ TM
        cmd_motor_speeds = cmd_rotor_thrusts / self.k_eta
        cmd_motor_speeds = np.sign(cmd_motor_speeds) * np.sqrt(np.abs(cmd_motor_speeds))

        # Assign controller commands. 
        cmd_thrust = u_new[0]                                             # Commanded thrust, in units N.
        cmd_moment = u_new[1:4]                                             # Commanded moment, in units N-m.
        cmd_q = baseline_control_input['cmd_q']              # Commanded attitude as a quaternion.
        cmd_w = baseline_control_input['cmd_w']
        # cmd_v = -self.kp_vel*pos_err + flat_output['x_dot']     # Commanded velocity in world frame (if using cmd_vel control abstraction), in units m/s
        cmd_v = flat_output['x_dot']     
        
        control_input = {'cmd_motor_speeds':cmd_motor_speeds.reshape(4,),
                         'cmd_motor_thrusts':cmd_rotor_thrusts,
                         'cmd_thrust':cmd_thrust[0],
                         'cmd_moment':cmd_moment.reshape(3,),
                         'cmd_q':cmd_q,
                         'cmd_w':cmd_w,
                         'cmd_v':cmd_v
                        }
        return control_input