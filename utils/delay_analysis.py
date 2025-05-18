import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt
from rotorpy.environments import Environment
from rotorpy.wind.default_winds import NoWind
from rotorpy.trajectories.random_motion_prim_traj import RapidTrajectory
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.circular_traj import CircularTraj

class DelayedControllerWrapper:
    """
    Wrapper for controllers that introduces a time delay in the control loop.
    """
    def __init__(self, controller, delay=0.1, buffer_size=1000):
        """
        Initialize the delayed controller wrapper.
        
        Parameters:
            controller: The controller to wrap
            delay: Time delay in seconds
            buffer_size: Size of the buffer for storing past control inputs
        """
        self.controller = controller
        self.delay = delay
        self.control_buffer = deque(maxlen=buffer_size)
        self.time_buffer = deque(maxlen=buffer_size)
        self.last_control = None
        
    def update(self, t, state, flat_output):
        """
        Update function with artificial delay.
        
        Parameters:
            t: Current time
            state: Current state
            flat_output: Desired flat outputs
            
        Returns:
            Delayed control input
        """
        # Compute current control input (but don't use it yet)
        current_control = self.controller.update(t, state, flat_output)
        
        # Store current control and time
        self.control_buffer.append(current_control)
        self.time_buffer.append(t)
        
        # If this is the first call, initialize last_control
        if self.last_control is None:
            self.last_control = current_control
            return current_control
        
        # Find the control input from delay seconds ago
        delayed_time = t - self.delay
        
        # If we don't have data from that far back, use the oldest available
        if len(self.time_buffer) == 0 or delayed_time < self.time_buffer[0]:
            return self.control_buffer[0]
        
        # Find the closest time in our buffer
        for i in range(len(self.time_buffer)-1, -1, -1):
            if self.time_buffer[i] <= delayed_time:
                return self.control_buffer[i]
        
        # Fallback to the most recent control input
        return self.last_control
    
    def update_trajectory(self, trajectory):
        """Pass through trajectory updates to the wrapped controller."""
        if hasattr(self.controller, 'update_trajectory'):
            self.controller.update_trajectory(trajectory)


def generate_random_trajectories(num_trajectories=5, traj_time=5.0):
    """
    Generate a set of random trajectories for testing.
    
    Parameters:
        num_trajectories: Number of trajectories to generate
        traj_time: Duration of each trajectory in seconds
        
    Returns:
        List of trajectory objects
    """
    trajectories = []
    trajectory_types = []
    
    # Add a hover trajectory
    trajectories.append(HoverTraj())
    trajectory_types.append("Hover")
    
    # Add a circular trajectory
    trajectories.append(CircularTraj(center=np.array([-2, 0, 0]), radius=2))
    trajectory_types.append("Circle")
    
    # Add random rapid trajectories
    for i in range(num_trajectories - 2):
        pos0 = np.array([0, 0, 0])
        vel0 = np.array([0, 0, 0])
        acc0 = np.array([0, 0, 0])
        gravity = np.array([0, 0, -9.81])
        
        trajectory = RapidTrajectory(pos0, vel0, acc0, gravity)
        
        # Generate random end conditions with increasing difficulty
        difficulty = (i + 1) / (num_trajectories - 2)  # Scale from 0 to 1
        
        # Position range increases with difficulty
        pos_range = 2.0 + 2.0 * difficulty
        posf = np.random.uniform(-pos_range, pos_range, size=3)
        posf[2] = abs(posf[2])  # Keep z positive
        
        # Velocity range increases with difficulty
        vel_range = 1.0 + 2.0 * difficulty
        velf = np.random.uniform(-vel_range, vel_range, size=3)
        
        # Acceleration range increases with difficulty
        acc_range = 0.5 + 1.5 * difficulty
        accf = np.random.uniform(-acc_range, acc_range, size=3)
        
        trajectory.set_goal_position(posf)
        trajectory.set_goal_velocity(velf)
        trajectory.set_goal_acceleration(accf)
        trajectory.generate(traj_time)
        
        trajectories.append(trajectory)
        trajectory_types.append(f"Rapid{i+1}")
    
    return trajectories, trajectory_types


def compute_delay_margin(controller_factory, controller_type, vehicle_params, trajectory, 
                         initial_delay=0.0, max_delay=0.5, delay_step=0.01,
                         test_duration=10.0, position_threshold=1.0):
    """
    Determine the delay margin by incrementally increasing delay until instability.
    
    Parameters:
        controller_factory: Function that creates a controller given type and params
        controller_type: String identifier for the controller
        vehicle_params: Parameters for the vehicle
        trajectory: Desired trajectory to follow
        initial_delay: Starting delay value (seconds)
        max_delay: Maximum delay to test (seconds)
        delay_step: Increment for delay testing (seconds)
        test_duration: Duration of each test simulation (seconds)
        position_threshold: Maximum position error before considering system unstable (meters)
        
    Returns:
        delay_margin: Maximum delay before instability (seconds)
        results: Dictionary of simulation results at different delays
    """
    from rotorpy.vehicles.multirotor import Multirotor
    
    results = {}
    delay_margin = max_delay  # Default if all tests pass
    
    # Create base controller
    base_controller = controller_factory(controller_type, vehicle_params)
    base_controller.update_trajectory(trajectory)
    
    # Create vehicle
    vehicle = Multirotor(vehicle_params)
    
    current_delay = initial_delay
    while current_delay <= max_delay:
        print(f"  Testing delay: {current_delay:.3f}s")
        
        # Create delayed controller
        delayed_controller = DelayedControllerWrapper(
            base_controller,
            delay=current_delay
        )
        
        # Set up simulation environment
        sim_env = Environment(
            vehicle=vehicle,
            controller=delayed_controller,
            trajectory=trajectory,
            wind_profile=NoWind(),
            sim_rate=100
        )
        
        # Initial state
        x0 = {
            'x': np.array([0, 0, 0]),
            'v': np.zeros(3,),
            'q': np.array([0, 0, 0, 1]),
            'w': np.zeros(3,),
            'wind': np.array([0,0,0]),
            'rotor_speeds': np.array([0,0,0,0])
        }
        
        # Run simulation
        result = sim_env.run(
            t_final=test_duration,
            use_mocap=False,
            terminate=False,
            plot=False,
            animate_bool=False,
            verbose=False
        )
        
        # Calculate maximum position error
        pos_error = np.linalg.norm(
            result['state']['x'] - result['flat']['x'], 
            axis=1
        )
        max_pos_error = np.max(pos_error)
        
        # Store results
        results[current_delay] = {
            'max_pos_error': max_pos_error,
            'unstable': max_pos_error > position_threshold,
            'result': result
        }
        
        # Check if system became unstable
        if max_pos_error > position_threshold:
            delay_margin = current_delay - delay_step
            print(f"  System became unstable at delay {current_delay:.3f}s (max error: {max_pos_error:.3f}m)")
            break
            
        current_delay += delay_step
    
    if current_delay > max_delay:
        print(f"  System remained stable for all tested delays up to {max_delay:.3f}s")
    
    return delay_margin, results


def plot_delay_margin_results(controller_types, delay_margins, detailed_results=None, save_path=None):
    """
    Plot delay margin comparison and stability boundaries.
    
    Parameters:
        controller_types: List of controller type strings
        delay_margins: List of delay margin values for each controller
        detailed_results: Optional dictionary of detailed results for each controller
        save_path: Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart of delay margins
    bars = ax1.bar(controller_types, delay_margins)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}',
                ha='center', va='bottom', rotation=0)
    
    ax1.set_ylabel('Delay Margin (seconds)')
    ax1.set_title('Maximum Tolerable Delay Before Instability')
    ax1.grid(True, axis='y')
    
    # If detailed results are provided, plot stability boundaries
    if detailed_results is not None:
        for i, ctrl_type in enumerate(controller_types):
            if ctrl_type in detailed_results:
                # Get the first trajectory's results (or any trajectory if it's a multi-trajectory test)
                if isinstance(detailed_results[ctrl_type], dict) and 'detailed_results' in detailed_results[ctrl_type]:
                    # Multi-trajectory format
                    first_traj = list(detailed_results[ctrl_type]['detailed_results'].keys())[0]
                    results = detailed_results[ctrl_type]['detailed_results'][first_traj]['results']
                else:
                    # Single trajectory format
                    results = detailed_results[ctrl_type]
                
                delays = sorted(list(results.keys()))
                errors = [results[d]['max_pos_error'] for d in delays]
                
                ax2.plot(delays, errors, 'o-', label=ctrl_type)
        
        ax2.axhline(y=1.0, color='r', linestyle='--', label='Stability Threshold')
        ax2.set_xlabel('Delay (seconds)')
        ax2.set_ylabel('Maximum Position Error (m)')
        ax2.set_title('Effect of Delay on Position Tracking')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Delay margin results saved to {save_path}")
    
    return fig


def visualize_delay_response(controller_type, results, save_path=None):
    """
    Visualize the system response at different delay values.
    
    Parameters:
        controller_type: String identifier for the controller
        results: Dictionary of simulation results at different delays
        save_path: Path to save the figure
    """
    delays = sorted(list(results.keys()))
    n_delays = len(delays)
    
    # Select a subset of delays to visualize if there are too many
    if n_delays > 4:
        # Choose stable, marginally stable, and unstable cases
        stable_delays = [d for d in delays if not results[d]['unstable']]
        unstable_delays = [d for d in delays if results[d]['unstable']]
        
        selected_delays = []
        if stable_delays:
            selected_delays.append(stable_delays[0])  # Smallest stable delay
            if len(stable_delays) > 1:
                selected_delays.append(stable_delays[-1])  # Largest stable delay
        
        if unstable_delays:
            selected_delays.append(unstable_delays[0])  # Smallest unstable delay
    else:
        selected_delays = delays
    
    fig, axes = plt.subplots(len(selected_delays), 2, figsize=(12, 4*len(selected_delays)))
    if len(selected_delays) == 1:
        axes = np.array([axes])
    
    for i, delay in enumerate(selected_delays):
        result = results[delay]['result']
        time = result['time']
        pos = result['state']['x']
        pos_des = result['flat']['x']
        
        # Position tracking
        for j in range(3):
            axes[i, 0].plot(time, pos[:, j], label=f'Actual {["x", "y", "z"][j]}')
            axes[i, 0].plot(time, pos_des[:, j], '--', label=f'Desired {["x", "y", "z"][j]}')
        
        axes[i, 0].set_title(f'Position Tracking (Delay = {delay:.3f}s)')
        axes[i, 0].set_xlabel('Time [s]')
        axes[i, 0].set_ylabel('Position [m]')
        if i == 0:
            axes[i, 0].legend()
        axes[i, 0].grid(True)
        
        # Position error
        pos_error = np.linalg.norm(pos - pos_des, axis=1)
        axes[i, 1].plot(time, pos_error)
        axes[i, 1].set_title(f'Position Error (Delay = {delay:.3f}s)')
        axes[i, 1].set_xlabel('Time [s]')
        axes[i, 1].set_ylabel('Error [m]')
        axes[i, 1].grid(True)
        
        # Add stability indicator
        if results[delay]['unstable']:
            axes[i, 1].text(0.5, 0.5, 'UNSTABLE', 
                           transform=axes[i, 1].transAxes,
                           fontsize=20, color='red', alpha=0.3,
                           ha='center', va='center')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Delay response visualization saved to {save_path}")
    
    return fig


def plot_multi_trajectory_results(controller_type, trajectory_types, delay_margins, save_path=None):
    """
    Plot delay margins across multiple trajectories for a single controller.
    
    Parameters:
        controller_type: String identifier for the controller
        trajectory_types: List of trajectory type strings
        delay_margins: List of delay margin values for each trajectory
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    bars = plt.bar(trajectory_types, delay_margins)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}',
                ha='center', va='bottom', rotation=0)
    
    plt.ylabel('Delay Margin (seconds)')
    plt.title(f'Delay Margins Across Different Trajectories for {controller_type}')
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi-trajectory results saved to {save_path}")
    
    return plt.gcf() 