""" Implementation of the nonlinear optimizer for the data-augmented MPC.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import scipy.io
import os
import sys
import shutil
import casadi as cs
import numpy as np
from copy import copy
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from controller.quadrotor_util import skew_symmetric, v_dot_q, safe_mkdir_recursive, quaternion_inverse, discretize_dynamics_and_cost

class QuadOptimizer:
    def __init__(self, quad_params, t_horizon=1, n_nodes=5,
                 q_cost=None, r_cost=None, q_mask=None,
                 model_name="quad_3d_acados_mpc", 
                 solver_options=None):
        """
        :param quad: quadrotor params
        :param t_horizon: time horizon for MPC optimization
        :param n_nodes: number of optimization nodes until time horizon
        :param q_cost: diagonal of Q matrix for LQR cost of MPC cost function. Must be a numpy array of length 12.
        :param r_cost: diagonal of R matrix for LQR cost of MPC cost function. Must be a numpy array of length 4.
        :param W_dnn: a matrix that maps the outputs of dnn to the state space.
        :param dnn: neural net model for correcting the nominal model
        :param q_mask: Optional boolean mask that determines which variables from the state compute towards the cost
        function. In case no argument is passed, all variables are weighted.
        :param solver_options: Optional set of extra options dictionary for solvers.
        :param rdrv_d_mat: 3x3 matrix that corrects the drag with a linear model according to Faessler et al. 2018. None
        if not used
        """

        # Load quad params
        self.quad_mass = quad_params['mass']
        Ixx = quad_params['Ixx']
        Iyy = quad_params['Iyy']
        Izz = quad_params['Izz']
        Ixy = quad_params['Ixy']
        Ixz = quad_params['Ixz']
        Iyz = quad_params['Iyz']
        self.J = np.array([[Ixx, Ixy, Ixz],
                           [Ixy, Iyy, Iyz],
                           [Ixz, Iyz, Izz]])
        self.inv_J = np.linalg.inv(self.J)

        k_m = quad_params['k_m']       # yaw moment coeff, Nm/(rad/s)**2
        k_eta = quad_params['k_eta']     # thrust coeff, N/(rad/s)**2
        k = k_m/k_eta
        num_rotors = quad_params['num_rotors']
        rotor_pos = quad_params['rotor_pos']
        self.f_to_TM = np.vstack((np.ones((1,num_rotors)),np.hstack([np.cross(rotor_pos[key],np.array([0,0,1])).reshape(-1,1)[0:2] for key in rotor_pos]), np.array([k*(-1)**i for i in range(num_rotors)]).reshape(1,-1)))
        self.TM_to_f = np.linalg.inv(self.f_to_TM)

        self.max_u = quad_params['rotor_speed_max']**2 * k_eta
        self.min_u = quad_params['rotor_speed_min']**2 * k_eta


        #load your own weight matrix Q and R for state and control input
        q_cost = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])
        r_cost = np.array([1,1,1,1])
        self.T = t_horizon  # Time horizon
        self.N = n_nodes  # number of control nodes within horizon

        # Declare model variables
        self.p = cs.MX.sym('x', 3)  # position
        self.v = cs.MX.sym('v', 3)  # velocity
        self.q = cs.MX.sym('q', 4)  # quaternion
        self.w = cs.MX.sym('w', 3)  # angular velocity

        # Full state vector (13-dimensional)
        self.x = cs.vertcat(self.p, self.v, self.q, self.w)
        self.state_dim = 13

        # Control input vector: motor thrust
        u1 = cs.MX.sym('u1')
        u2 = cs.MX.sym('u2')   
        u3 = cs.MX.sym('u3') 
        u4 = cs.MX.sym('u4') 
        self.u = cs.vertcat(u1, u2, u3, u4)

        # Nominal model equations symbolic function (no augmented dynamics)
        self.quad_xdot_nominal = self.quad_dynamics()

        # Initialize objective function, 0 target state and integration equations
        self.L = None
        self.target = None

        # Set up acados model
        acados_models, nominal_with_dnn = self.acados_setup_model(
            self.quad_xdot_nominal(x=self.x, u=self.u)['x_dot'], model_name)

        # ### Setup and compile Acados OCP solvers ### #
        self.acados_ocp_solver = {}

        # # Add one more weight to the rotation (use quaternion norm weighting in acados)
        # q_diagonal = np.concatenate((q_cost[:6], np.mean(q_cost[6:9])[np.newaxis], q_cost[6:]))
        # if q_mask is not None:
        #     q_mask = np.concatenate((q_mask[:6], np.zeros(1), q_mask[6:]))
        #     q_diagonal *= q_mask

        # Ensure current working directory is current folder
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        self.acados_models_dir = '../../acados_models'
        safe_mkdir_recursive(os.path.join(os.getcwd(), self.acados_models_dir))

        print(self.acados_models_dir)

        for key, key_model in zip(acados_models.keys(), acados_models.values()):

            nx = key_model.x.size()[0]
            nu = key_model.u.size()[0]
            ny = nx + nu

            if isinstance(key_model.p, cs.MX):
                n_param = key_model.p.size()[0]
            else:
                n_param = 0

            acados_source_path = os.environ['ACADOS_SOURCE_DIR']
            sys.path.insert(0, '../common')

            # Create OCP object to formulate the optimization
            ocp = AcadosOcp()
            ocp.acados_include_path = acados_source_path + '/include'
            ocp.acados_lib_path = acados_source_path + '/lib'
            ocp.model = key_model
            ocp.dims.N = self.N
            ocp.solver_options.tf = t_horizon

            # Initialize parameters
            ocp.dims.np = n_param
            ocp.parameter_values = np.zeros(n_param)

            ocp.cost.cost_type = 'LINEAR_LS'
            ocp.cost.cost_type_e = 'LINEAR_LS'

            ocp.cost.W = np.diag(np.concatenate((q_cost, r_cost)))
            ocp.cost.W_e = ocp.cost.W[0:nx,0:nx];
            # terminal_cost = 0 if solver_options is None or not solver_options["terminal_cost"] else 1
            # ocp.cost.W_e *= terminal_cost

            ocp.cost.Vx = np.zeros((ny, nx))
            ocp.cost.Vx[:nx, :nx] = np.eye(nx)
            ocp.cost.Vu = np.zeros((ny, nu))
            ocp.cost.Vu[-4:, -4:] = np.eye(nu)

            ocp.cost.Vx_e = np.eye(nx)

            # Initial reference trajectory (will be overwritten)
            x_ref = np.zeros(nx)
            ocp.cost.yref = np.concatenate((x_ref, np.array([0.0, 0.0, 0.0, 0.0])))
            ocp.cost.yref_e = x_ref

            # Initial state (will be overwritten)
            ocp.constraints.x0 = x_ref

            # Set constraints
            ocp.constraints.lbu = np.array([self.min_u] * 4)
            ocp.constraints.ubu = np.array([self.max_u] * 4)
            ocp.constraints.idxbu = np.array([0, 1, 2, 3])

            # Solver options
            ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
            ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
            ocp.solver_options.integrator_type = 'ERK'
            ocp.solver_options.print_level = 0
            ocp.solver_options.nlp_solver_type = 'SQP_RTI' if solver_options is None else solver_options["solver_type"]

            # Compile acados OCP solver if necessary
            json_file = os.path.join(self.acados_models_dir, key_model.name + '_acados_ocp.json')
            self.acados_ocp_solver[key] = AcadosOcpSolver(ocp, json_file=json_file)

    def clear_acados_model(self):
        """
        Removes previous stored acados models to avoid name conflicts.
        """

        json_file = os.path.join(self.acados_models_dir, 'acados_ocp.json')
        if os.path.exists(json_file):
            os.remove(os.path.join(os.getcwd(), json_file))
        compiled_model_dir = os.path.join(os.getcwd(), 'c_generated_code')
        if os.path.exists(compiled_model_dir):
            shutil.rmtree(compiled_model_dir)

    def add_missing_states(self, dnn_outs):
        """
        Adds 0's in case the DNN's only regressed a subset of the states.
        """
        # states predicted by the DNN. 0 if empty 
        output = cs.vertcat(cs.MX.zeros(3), dnn_outs[:3], cs.MX.zeros(4), cs.MX.zeros(3))
        return output

    def remove_extra_states(self, x):
        """
        remove extra states not used by DNN
        """
        dnn_input_x = x[3:]
        return dnn_input_x

    def acados_setup_model(self, nominal, model_name):
        """
        Builds an Acados symbolic models using CasADi expressions.
        :param model_name: name for the acados model. Must be different from previously used names or there may be
        problems loading the right model.
        :param nominal: CasADi symbolic nominal model of the quadrotor: f(self.x, self.u) = x_dot, dimensions 13x1.
        :return: Returns a total of three outputs, where m is the number of GP's in the GP ensemble, or 1 if no DNN:
            - A dictionary of m AcadosModel of the GP-augmented quadrotor
            - A dictionary of m CasADi symbolic nominal dynamics equations with GP mean value augmentations (if with GP)
        :rtype: dict, dict, cs.MX
        """

        def fill_in_acados_model(x, u, p, dynamics, name):

            x_dot = cs.MX.sym('x_dot', dynamics.shape)
            f_impl = x_dot - dynamics

            # Dynamics model
            model = AcadosModel()
            model.f_expl_expr = dynamics
            model.f_impl_expr = f_impl
            model.x = x
            model.xdot = x_dot
            model.u = u
            model.p = p
            model.name = name

            return model

        acados_models = {}
        dynamics_equations = {}

        # No available DNN model so return nominal dynamics
        dynamics_equations[0] = nominal
        x_ = self.x
        dynamics_ = nominal
        acados_models[0] = fill_in_acados_model(x=x_, u=self.u, p=[], dynamics=dynamics_, name=model_name)

        return acados_models, dynamics_equations

    def quad_dynamics(self):
        """
        Symbolic dynamics of the 3D quadrotor model. The state consists on: [p_xyz, a_wxyz, v_xyz, r_xyz]^T, where p
        stands for position, a for angle (in quaternion form), v for velocity and r for body rate. The input of the
        system is: [u_1, u_2, u_3, u_4], i.e. the activation of the four thrusters.

        :param rdrv_d: a 3x3 diagonal matrix containing the D matrix coefficients for a linear drag model as proposed
        by Faessler et al.

        :return: CasADi function that computes the analytical differential state dynamics of the quadrotor model.
        Inputs: 'x' state of quadrotor (6x1) and 'u' control input (2x1). Output: differential state vector 'x_dot'
        (6x1)
        """
        x_dot = cs.vertcat(self.x_dynamics(), self.v_dynamics(), self.q_dynamics(), self.w_dynamics())
        return cs.Function('x_dot', [self.x[:13], self.u], [x_dot], ['x', 'u'], ['x_dot'])

    def x_dynamics(self):
        return self.v

    def q_dynamics(self):
        return 1 / 2 * cs.mtimes(skew_symmetric(self.w), self.q)

    def v_dynamics(self):
        f_thrust = self.u
        g = cs.vertcat(0.0, 0.0, 9.81)
        ind = True
        if ind:
            a_thrust = cs.vertcat(0.0, 0.0, f_thrust[0]+f_thrust[1]+f_thrust[2]+f_thrust[3]) / self.quad_mass
        else:
            a_thrust = cs.vertcat(0.0, 0.0, f_thrust[0]) / self.quad_mass
        v_dynamics = v_dot_q(a_thrust, self.q) - g

        return v_dynamics

    def w_dynamics(self):
        f_thrust = self.u
        ind = True
        if ind:
            TM = cs.mtimes(self.f_to_TM, f_thrust)
            tau_b = cs.vertcat(TM[1], TM[2], TM[3])
        else:
            tau_b = cs.vertcat(f_thrust[1], f_thrust[2], f_thrust[3])
        tmp = tau_b - cs.cross(self.w, cs.mtimes(self.J, self.w))
        return cs.mtimes(self.inv_J, tmp)

    def set_reference_state(self, x_target=None, u_target=None):
        """
        Sets the target state and pre-computes the integration dynamics with cost equations
        :param x_target: 13-dimensional target state (p_xyz, a_wxyz, v_xyz, r_xyz)
        :param u_target: 4-dimensional target control input vector (u_1, u_2, u_3, u_4)
        """

        if x_target is None:
            x_target = [[0, 0, 0.0], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0]]
        if u_target is None:
            u_target = [0, 0, 0, 0]

        # Set new target state
        self.target = copy(x_target)

        ref = np.concatenate([x_target[i] for i in range(4)])
        #  Transform velocity to body frame
        v_b = v_dot_q(ref[3:6], quaternion_inverse(ref[6:10]))
        ref = np.concatenate((ref[:3], v_b, ref[6:]))

        # Determine which dynamics model to use based on the GP optimal input feature region. Should get one for each
        # output dimension of the GP
        gp_ind = 0

        ref = np.concatenate((ref, u_target))

        for j in range(self.N):
            self.acados_ocp_solver[gp_ind].set(j, "yref", ref)
        self.acados_ocp_solver[gp_ind].set(self.N, "yref", ref[:-4])

        return gp_ind

    def set_reference_trajectory(self, x_target, u_target, use_model=0):
        """
        Sets the reference trajectory and pre-computes the cost equations for each point in the reference sequence.
        :param x_target: Nx13-dimensional reference trajectory (p_xyz, v_xyz, angle_wxyz, rate_xyz). It is passed in the
        form of a 4-length list, where the first element is a Nx3 numpy array referring to the position targets, the
        second is a Nx4 array referring to the quaternion, two more Nx3 arrays for the velocity and body rate targets.
        :param u_target: Nx4-dimensional target control input vector (u1, u2, u3, u4)
        """

        if u_target is not None:
            assert x_target[0].shape[0] == (u_target.shape[0] + 1) or x_target[0].shape[0] == u_target.shape[0]

        # If not enough states in target sequence, append last state until required length is met
        while x_target[0].shape[0] < self.N + 1:
            x_target = [np.concatenate((x, np.expand_dims(x[-1, :], 0)), 0) for x in x_target]
            if u_target is not None:
                u_target = np.concatenate((u_target, np.expand_dims(u_target[-1, :], 0)), 0)

        stacked_x_target = np.concatenate([x for x in x_target], 1)

        #  Transform velocity to body frame
        x_mean = stacked_x_target[int(self.N / 2)]
        v_b = v_dot_q(x_mean[3:6], quaternion_inverse(x_mean[6:10]))
        x_target_mean = np.concatenate((x_mean[:3], v_b, x_mean[6:]))

        self.target = copy(x_target)

        for j in range(self.N):
            ref = stacked_x_target[j, :]
            ref = np.concatenate((ref, u_target[j, :]))
            self.acados_ocp_solver[use_model].set(j, "yref", ref)
        # the last MPC node has only a state reference but no input reference
        self.acados_ocp_solver[use_model].set(self.N, "yref", stacked_x_target[self.N, :])
        return use_model

    def run_optimization(self, initial_state=None, use_model=0, return_x=False, task_index=0):
        """
        Optimizes a trajectory to reach the pre-set target state, starting from the input initial state, that minimizes
        the quadratic cost function and respects the constraints of the system

        :param initial_state: 13-element list of the initial state. If None, 0 state will be used
        :param use_model: integer, select which model to use from the available options.
        :param return_x: bool, whether to also return the optimized sequence of states alongside with the controls.
        :param gp_regression_state: 13-element list of state for GP prediction. If None, initial_state will be used.
        :return: optimized control input sequence (flattened)
        """

        if initial_state is None:
            initial_state = [0, 0, 0] + [0, 0, 0] + [1, 0, 0, 0] + [0, 0, 0]

        # Set initial state. Add gp state if needed
        x_init = initial_state
        x_init = np.stack(x_init)

        # Set initial condition, equality constraint
        self.acados_ocp_solver[use_model].set(0, 'lbx', x_init)
        self.acados_ocp_solver[use_model].set(0, 'ubx', x_init)

        # Solve OCP
        self.acados_ocp_solver[use_model].solve()


        # Solve du/dx_init
        sens_u = np.zeros((4,13))
        field = 'ex'
        stage = 0
        for index in range(13):
            self.acados_ocp_solver[use_model].eval_param_sens(index,stage,field)
            temp = self.acados_ocp_solver[use_model].get(0,'sens_u')
            sens_u[:,index] = temp
        
        
        # Get u, N is number of steps in MPC horizon
        w_opt_acados = np.ndarray((self.N, 4))
        x_opt_acados = np.ndarray((self.N + 1, len(x_init)))
        x_opt_acados[0, :] = self.acados_ocp_solver[use_model].get(0, "x")
        for i in range(self.N):
            w_opt_acados[i, :] = self.acados_ocp_solver[use_model].get(i, "u")
            x_opt_acados[i + 1, :] = self.acados_ocp_solver[use_model].get(i + 1, "x")

        w_opt_acados = np.reshape(w_opt_acados, (-1))
        return w_opt_acados if not return_x else (w_opt_acados, x_opt_acados,sens_u)

