import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib import rcParams
from quad_param.quadrotor import quad_params



def ModifyPlotForPublication(text_size=10, tick_size=10, legend_size=10):
    """Set matplotlib parameters for publication quality plots"""
    rcParams['axes.labelsize'] = text_size
    rcParams['axes.formatter.use_mathtext'] = True
    rcParams['xtick.labelsize'] = tick_size
    rcParams['ytick.labelsize'] = tick_size
    rcParams['legend.fontsize'] = legend_size
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['cmr10']
    rcParams['text.usetex'] = True

def plot_3d_trajectory(ax, sim_results, controller_types, controller_palette,
                       elev=20, azim=-75,roll=0):
    """Plot 3D trajectory comparison"""
    x_des = sim_results[0]['flat']['x']
    ax.plot(x_des[:,0], x_des[:,1], x_des[:,2], 'k--', label='Desired')
    ax.scatter(x_des[0,0], x_des[0,1], x_des[0,2], color='red', label='Start')
    ax.scatter(x_des[-1,0], x_des[-1,1], x_des[-1,2], color='blue', label='End')
    
    for result, ctrl_type in zip(sim_results, controller_types):
        x = result['state']['x']
        ax.plot(x[:,0], x[:,1], x[:,2], label=ctrl_type, 
                color=controller_palette[controller_types.index(ctrl_type)],linewidth=2)
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_zlim(-2,2)
    # ax.set_title('3D Trajectory')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.legend(loc='best')

def plot_position_error(ax, sim_results, controller_types, controller_palette):
    """Plot position tracking error norm for all controllers"""
    for result, ctrl_type in zip(sim_results, controller_types):
        time = result['time']
        x = result['state']['x']
        x_des = result['flat']['x']
        pos_error = np.linalg.norm(x - x_des, axis=1)
        ax.plot(time, pos_error, label=ctrl_type, 
                color=controller_palette[controller_types.index(ctrl_type)])
    
    ax.set_ylabel('$\| \mathbf{x} - \mathbf{x}_{des} \|_2$ [m]')
    ax.legend()
    ax.grid(True)

def plot_disturbance(ax, result, experiment_type, disturbance_palette):
    """Plot disturbance data"""
    time = result['time']
    if experiment_type == 'wind':
        wind = result['state']['wind']
        ax.plot(time, wind[:,0], label='X', color=disturbance_palette[0])
        ax.plot(time, wind[:,1], label='Y', color=disturbance_palette[1])
        ax.plot(time, wind[:,2], label='Z', color=disturbance_palette[2])
        ax.set_ylabel('Wind [m/s]')
    elif experiment_type == 'torque':
        torque = result['state']['ext_torque']
        ax.plot(time, torque[:,0], label='X', color=disturbance_palette[0])
        ax.plot(time, torque[:,1], label='Y', color=disturbance_palette[1])
        ax.plot(time, torque[:,2], label='Z', color=disturbance_palette[2])
        ax.set_ylabel('Torque [Nm]')
    elif experiment_type == 'force':
        force = result['state']['ext_force']
        ax.plot(time, force[:,0], label='X', color=disturbance_palette[0])
        ax.plot(time, force[:,1], label='Y', color=disturbance_palette[1])
        ax.plot(time, force[:,2], label='Z', color=disturbance_palette[2])
        ax.set_ylabel('Force [N]')
    else:
        raise ValueError(f"This experiment type cannot be visualized by this generalized plotting function: {experiment_type}")
    ax.legend()
    ax.grid(True)

def plot_motor_speeds(ax, sim_results, controller_types, controller_palette):
    """Plot average motor speeds"""
    for result, ctrl_type in zip(sim_results, controller_types):
        motor_speeds = result['control']['cmd_motor_speeds']
        time = result['time']
        ax.plot(time, motor_speeds.mean(axis=1), label=ctrl_type, 
                color=controller_palette[controller_types.index(ctrl_type)])
    
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Average Motor Speed [rad/s]')
    ax.legend()
    ax.grid(True)

def get_position_error_range(sim_results):
    """Calculate global min/max position error"""
    pos_error_min = float('inf')
    pos_error_max = float('-inf')
    for result in sim_results:
        pos_error = result['state']['x'] - result['flat']['x']
        pos_error_min = min(pos_error_min, pos_error.min())
        pos_error_max = max(pos_error_max, pos_error.max())
    return pos_error_min, pos_error_max

def get_rotor_speed_range(sim_results):
    """Calculate global min/max rotor speeds"""
    speed_min = float('inf')
    speed_max = float('-inf')
    for result in sim_results:
        speeds = result['state']['rotor_speeds']
        speed_min = min(speed_min, speeds.min())
        speed_max = max(speed_max, speeds.max())
    return speed_min, speed_max

def plot_position_errors_subplot(fig, pos, result, ctrl_type, is_leftmost, ymin, ymax):
    """Plot position errors for a single controller"""
    ax = fig.add_subplot(pos)
    pos_palette = sns.color_palette()
    time = result['time']
    pos_error = result['state']['x'] - result['flat']['x']
    
    for i, label in enumerate(['X', 'Y', 'Z']):
        ax.plot(time, pos_error[:,i], label=label, color=pos_palette[i])
    
    ax.set_title(f'{ctrl_type}')
    if is_leftmost:
        ax.set_ylabel('Position Error [m]')
    ax.legend()
    ax.grid(True)
    ax.set_ylim(ymin, ymax)

def plot_rotor_speeds_subplot(fig, pos, result, ctrl_type, is_leftmost, ymin, ymax, window):
    """Plot rotor speeds for a single controller"""
    ax = fig.add_subplot(pos)
    rotor_palette = sns.color_palette("husl", 4)
    time = result['time']
    rotor_speeds = result['state']['rotor_speeds']

    window_size = 20
    weights = np.repeat(1.0, window_size)/window_size
    
    for i in range(4):
        smoothed = np.convolve(rotor_speeds[:,i], window, mode='same')
        # Scale the data to motor speed range
        ax.plot(time, smoothed, label=f'Rotor {i+1}')
        # ax.plot(time, rotor_speeds[:,i], label=f'Rotor {i+1}', color=rotor_palette[i])
    
    if is_leftmost:
        ax.set_ylabel('Rotor Speed [rad/s]')
    ax.legend()
    ax.grid(True)
    ax.set_ylim(ymin, ymax) 

def plot_drone(ax):
    # Draw quadrotor configuration
    arm_length = quad_params['arm_length']
    
    # Get rotor positions from quad_params
    rotor_positions = [
        quad_params['rotor_pos']['r1'][:2],  # front left
        quad_params['rotor_pos']['r2'][:2],  # front right
        quad_params['rotor_pos']['r3'][:2],  # back right
        quad_params['rotor_pos']['r4'][:2],  # back left
    ]
    
    # Draw arms connecting diagonal rotors
    ax.plot([rotor_positions[0][0], rotor_positions[2][0]], 
                [rotor_positions[0][1], rotor_positions[2][1]], 'k-', linewidth=2)  # FL to BR
    ax.plot([rotor_positions[1][0], rotor_positions[3][0]], 
                [rotor_positions[1][1], rotor_positions[3][1]], 'k-', linewidth=2)  # FR to BL
    
    # Draw rotors with direction indicators
    rotor_directions = quad_params['rotor_directions']
    for i, (pos, direction) in enumerate(zip(rotor_positions, rotor_directions)):
        # Plot rotor position
        ax.plot(pos[0], pos[1], 'ko', markersize=8)
        
        # Add rotor labels
        ax.annotate(f'mot{i}', pos, xytext=(5, 5), textcoords='offset points')
    # Set equal axis limits centered on zero
    limit = 1.5 * quad_params['arm_length']
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)


def plot_model_uncertainty(ax,controller_param, vehicle_params):
        """Specific plotting for model uncertainty experiments"""
        params_to_plot = [
            'mass', 'Ixx', 'Iyy', 'Izz', 'arm_length',
            'k_eta', 'k_m','rotor_speed_max'
        ]
        params_to_label = [
            '$m$', '$\mathbf{J}_{xx}$', '$\mathbf{J}_{yy}$', '$\mathbf{J}_{zz}$', '$l$',
            '$k_{t}$', '$k_{\tau}$', '$\omega_{\max}$'
        ]
        
        # Normalize values for comparison
        values_gt = []
        values_ref = []
        
        for param in params_to_plot:
            gt_val = vehicle_params[param]
            ref_val = controller_param[param]
            # Normalize relative to reference value
            values_gt.append(gt_val / ref_val)
            values_ref.append(1.0)  # Reference is always 1.0 after normalization
    
        # Set up the angles for each parameter (evenly spaced)
        angles = np.linspace(0, 2*np.pi, len(params_to_plot), endpoint=False)
        
        # Close the plot by appending first values
        values_gt = np.concatenate((values_gt, [values_gt[0]]))
        values_ref = np.concatenate((values_ref, [values_ref[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        params_to_plot = np.concatenate((params_to_plot, [params_to_plot[0]]))
        params_to_label = np.concatenate((params_to_label,[params_to_label[0]]))
        
        # Plot data
        ax.plot(angles, values_gt, 'o-', linewidth=1.5, label='Ground Truth')
        ax.plot(angles, values_ref, 'o-', linewidth=1.5, label='Reference')
        ax.fill(angles, values_gt, alpha=0.25)
        ax.fill(angles, values_ref, alpha=0.25)
        
        # Set the labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(params_to_label[:-1])
        ax.legend()
        ax.set_title('Model Parameter Comparison \n(Normalized to Reference Values)', pad=30)  # Increase pad value if needed
