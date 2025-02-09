import numpy as np
# from rotorpy.controllers.quadrotor_traopt import QuadOptimizer
from controller.quadrotor_traopt import QuadOptimizer
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.vehicles.hummingbird_params import quad_params  # There's also the Hummingbird
from rotorpy.trajectories.circular_traj  import CircularTraj, ThreeDCircularTraj
from rotorpy.trajectories.hover_traj  import HoverTraj
from rotorpy.controllers.quadrotor_util import minimum_snap_trajectory_generator
from rotorpy.controllers.quadrotor_dnn import DNN_model, IB_model

class QuadMPC:
    def __init__(self, quad_params=quad_params, trajectory=CircularTraj(),
                 t_final=10, t_horizon=1, n_nodes=10,
                 q_cost=None, r_cost=None, q_mask=None, model_name='quad_3d_acados_mpc', solver_options=None, plot_traj=False):
        """

        """
            
        self.quad_opt = QuadOptimizer(quad_params=quad_params, t_horizon=t_horizon, n_nodes=n_nodes,
                                      q_cost=q_cost, r_cost=r_cost, q_mask=q_mask, 
                                      model_name=model_name, solver_options=solver_options)

        plot = plot_traj
        self.x_ref_list, self.u_ref_list, self.t_ref = self.prepare_ref_traj(trajectory, t_final, t_horizon, n_nodes, plot)

    def prepare_ref_traj(self, traj, t_final, t_horizon, n_nodes, plot=False):
        N = int(t_final/t_horizon * n_nodes)
        t_ref = np.linspace(0, t_final, N)
        traj_d = np.zeros((4,3,N))
        yaw_d = np.zeros((2,N))
        for i,t in enumerate(t_ref):
            state = traj.update(t)
            traj_d[0,:,i] = state['x']
            traj_d[1,:,i] = state['x_dot']
            traj_d[2,:,i] = state['x_ddot']
            traj_d[3,:,i] = state['x_dddot']
            yaw_d[0,i] = state['yaw']
            yaw_d[1,i] = state['yaw_dot']
        x_ref, t_ref, u_ref = minimum_snap_trajectory_generator(traj_d, yaw_d, t_ref, quad_params, 
                                                                   map_limits=None, plot=plot, to_list=True)
        x_ref_list = []
        u_ref_list = []
        for i_l in range(N):
            i_r = int(i_l + n_nodes)
            if i_l > N - n_nodes:
                x_ref_list.append([x_ref[0][i_l:,:], x_ref[1][i_l:,:], x_ref[2][i_l:,:], x_ref[3][i_l:,:]])
                u_ref_list.append(u_ref[i_l:,:])
            else:
                x_ref_list.append([x_ref[0][i_l:i_r,:], x_ref[1][i_l:i_r,:], x_ref[2][i_l:i_r,:], x_ref[3][i_l:i_r,:]])
                u_ref_list.append(u_ref[i_l:i_r,:])
        return x_ref_list, u_ref_list, t_ref
        
    def set_reference(self, index):
        """
        t, present time in seconds
        index, index of reference trajectory in the list
        """
        if index >= len(self.x_ref_list):
            index = len(self.x_ref_list) - 1
        x_ref = self.x_ref_list[index]
        u_ref = self.u_ref_list[index]

        use_model = 0

        self.quad_opt.set_reference_trajectory(x_target=x_ref, u_target=u_ref, use_model=use_model)

    def run_optimization(self, initial_state=None, return_x=True, task_index=None):
        x_init = initial_state
        use_model = 0

        w_opt,x_opt,sens_u = self.quad_opt.run_optimization(initial_state=x_init, use_model=use_model, return_x=return_x)
        return w_opt,x_opt,sens_u

    def clear_model(self):
        self.quad_opt.clear_acados_model()

    
