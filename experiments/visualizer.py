import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from experiments.plotting_utils import *
from utils.parallel_data_collection import compute_cost
from config.randomization_config import ExperimentType
from rotorpy.vehicles.multirotor import Multirotor
@dataclass
class VisualizationConfig:
    """Configuration for experiment visualization"""
    sim_results: List[Dict[str, Any]]
    controller_types: List[str]
    controller_palette: Optional[Any] = None
    disturbance_palette: Optional[Any] = None
    controller_param: Optional[Dict] = None
    vehicle_params: Optional[Dict] = None
    vehicle: Optional[Multirotor] = None
    experiment_type: Optional[str] = None

class ExperimentVisualizer:
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / 'data'
        self.plot_dir = self.data_dir / 'plots'
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        ModifyPlotForPublication()

    def visualize_trials(self, experiment_type, sim_results, controller_types, controller_param, vehicle_params, vehicle):
        """Create plots based on experiment type
        
        Args:
            experiment_type (str): Type of experiment ('wind', 'force', 'torque', 'uncertainty', 'rotoreff', 'payload')
            sim_results (list): List of simulation results
            controller_types (list): List of controller names
            controller_param (list): List of controller parameters
            vehicle_params (dict): Vehicle parameters
            vehicle (Multirotor): Vehicle object
        Raises:
            ValueError: If experiment_type is not supported
        """
        # Create visualization config
        config = VisualizationConfig(
            sim_results=sim_results,
            controller_types=controller_types,
            controller_param=controller_param,
            vehicle_params=vehicle_params,
            vehicle=vehicle,
            controller_palette=sns.color_palette("husl", len(controller_types)),
            disturbance_palette=sns.color_palette("Set1", 3),
            experiment_type=experiment_type
        )
        
        plot_functions = {
            'wind': self._plot_wind_experiment,
            'force': self._plot_force_experiment,
            'torque': self._plot_torque_experiment,
            'uncertainty': self._plot_uncertainty_experiment,
            'rotoreff': self._plot_rotor_efficiency_experiment,
            'payload': self._plot_payload_experiment
        }
        
        if experiment_type not in plot_functions:
            raise ValueError(f"Unsupported experiment type: {experiment_type}. "
                            f"Must be one of {list(plot_functions.keys())}")
        
        fig = plot_functions[experiment_type](config)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / f'vis_{experiment_type}_{"_".join(controller_types)}.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()

        self._print_metrics(sim_results, controller_types)
        
        return sim_results

    def _plot_wind_experiment(self, config: VisualizationConfig):
        """Specific plotting for wind experiments"""
        fig = plt.figure(figsize=(8,4))
        gs = fig.add_gridspec(2,4)
        
        # 3D trajectory and wind vector plot
        ax1 = fig.add_subplot(gs[0:, 0:2], projection='3d')
        plot_3d_trajectory(ax1, config.sim_results, config.controller_types, config.controller_palette)
        
        # Wind components
        ax2 = fig.add_subplot(gs[1, 2:])
        plot_disturbance(ax2, config.sim_results[0], 'wind', config.disturbance_palette)
        ax2.set_ylabel('Wind [m/s]')
        # Position tracking error
        ax3 = fig.add_subplot(gs[0, 2:])
        plot_position_error(ax3, config.sim_results, config.controller_types, config.controller_palette)
        
        return fig
    
    def _plot_force_experiment(self, config: VisualizationConfig):
        """Specific plotting for force experiments"""
        fig = plt.figure(figsize=(8,4))
        gs = fig.add_gridspec(2,4)
        
        # 3D trajectory and force vector plot
        ax1 = fig.add_subplot(gs[0:, 0:2], projection='3d')
        plot_3d_trajectory(ax1, config.sim_results, config.controller_types, config.controller_palette)
        
        # Force components
        ax2 = fig.add_subplot(gs[1, 2:])
        plot_disturbance(ax2, config.sim_results[0], 'force', config.disturbance_palette)
        ax2.set_ylabel('Force [N]')
        # Position tracking error
        ax3 = fig.add_subplot(gs[0, 2:])
        plot_position_error(ax3, config.sim_results, config.controller_types, config.controller_palette)
        
        return fig
    
    def _plot_torque_experiment(self, config: VisualizationConfig):
        """Specific plotting for torque experiments"""
        fig = plt.figure(figsize=(8,4))
        gs = fig.add_gridspec(2,4)
        
        # 3D trajectory and torque vector plot
        ax1 = fig.add_subplot(gs[0:, 0:2], projection='3d')
        plot_3d_trajectory(ax1, config.sim_results, config.controller_types, config.controller_palette)
        
        # Torque components
        ax2 = fig.add_subplot(gs[1, 2:])
        plot_disturbance(ax2, config.sim_results[0], 'torque', config.disturbance_palette)
        ax2.set_ylabel('Torque [Nm]')
        # Position tracking error
        ax3 = fig.add_subplot(gs[0, 2:])
        plot_position_error(ax3, config.sim_results, config.controller_types, config.controller_palette)

        return fig

    def _plot_rotor_efficiency_experiment(self, config: VisualizationConfig):
        num_controllers = len(config.controller_types)
        fig = plt.figure(figsize=(8,6*num_controllers))
        each_row_num = 3
        gs = fig.add_gridspec(each_row_num*num_controllers, 4)

        # Get global min/max values
        pos_error_min, pos_error_max = get_position_error_range(config.sim_results)
        rotor_speed_min, rotor_speed_max = get_rotor_speed_range(config.sim_results)
        window = np.ones(20)/20  # 20-point moving average

        for idx, (result, ctrl_type) in enumerate(zip(config.sim_results, config.controller_types)):
            ax1 = fig.add_subplot(gs[idx*each_row_num:(idx+1)*each_row_num, 0:2], projection='3d')
            controller_palette = sns.color_palette("husl", len(config.controller_types))
            plot_3d_trajectory(ax1, config.sim_results, config.controller_types, controller_palette)
            plot_position_errors_subplot(fig, gs[idx*each_row_num, 2:], result, ctrl_type, idx==0,
                                    pos_error_min, pos_error_max)
            plot_rotor_speeds_subplot(fig, gs[idx*each_row_num+1,2:], result, ctrl_type, idx==0,
                                    rotor_speed_min, rotor_speed_max, window)
            # plot the rotor efficiency
            ax3 = fig.add_subplot(gs[idx*each_row_num+2, 2:])
            rotor_eff = np.tile(config.vehicle_params['rotor_efficiency'], (result['time'].shape[0], 1))
            ax3.plot(result['time'], rotor_eff[:,0], label='Rotor 1')
            ax3.plot(result['time'], rotor_eff[:,1], label='Rotor 2')
            ax3.plot(result['time'], rotor_eff[:,2], label='Rotor 3')
            ax3.plot(result['time'], rotor_eff[:,3], label='Rotor 4')
            ax3.legend()
            ax3.grid(True)
            ax3.set_xlabel('Time [s]')
            ax3.set_ylabel('Rotor Efficiency')
            
        return fig
    
    def _plot_uncertainty_experiment(self, config: VisualizationConfig):
        """Specific plotting for uncertainty experiments"""
        """
        Create a radar chart comparing ground truth and reference model parameters
        
        Args:
            ground_truth_params (dict): Ground truth parameters
            reference_params (dict): Reference model parameters
        """
        # Select key parameters to compare
        fig = plt.figure(figsize=(8,4))
        gs = fig.add_gridspec(2,4)
        ax1 = fig.add_subplot(gs[0:, 0:2], projection='3d')
        plot_3d_trajectory(ax1, config.sim_results, config.controller_types, config.controller_palette)
        ax1.set_title('3D Trajectory')
        ax2 = fig.add_subplot(gs[0:, 2:], projection='polar')
        plot_model_uncertainty(ax2, config.controller_param, config.vehicle_params)

        return fig
    
    def _plot_payload_experiment(self, config: VisualizationConfig):
        """Specific plotting for payload experiments"""
        fig = plt.figure(figsize=(6,6))
        gs = fig.add_gridspec(4,4)
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(gs[0:2, 0:2], projection='3d')
        plot_3d_trajectory(ax1, config.sim_results, config.controller_types, 
                           config.controller_palette, elev=45, azim=-90, roll=0)
        
        # Position tracking error
        ax2 = fig.add_subplot(gs[2, :])
        plot_position_error(ax2, config.sim_results, config.controller_types, config.controller_palette)
        
        # Payload mass ratio and attachment visualization
        print(config.sim_results[0]['state'].keys())
        ext_torque = config.sim_results[0]['state']['ext_torque']

        time = config.sim_results[0]['time']
        
        # Calculate when the payload is attached (ext_torque is non-zero)
        payload_attached = np.sum(np.abs(ext_torque), axis=1) > 0.01  # Small threshold to handle numerical precision
        payload_ratio_array = np.zeros_like(time)
        payload_ratio_array[payload_attached] = config.vehicle.payload_mass / config.vehicle.mass
        
        # Plot payload mass ratio
        ax3 = fig.add_subplot(gs[3, :])
        ax3.plot(time, payload_ratio_array)
        ax3.set_ylabel('Payload Mass Ratio')
        ax3.grid(True)

        # Payload location and COM visualization (top-down view)
        ax4 = fig.add_subplot(gs[0:2, 2:], aspect='equal')
        plot_drone(ax4)
        
        # Plot the COM position
        com = config.vehicle.com
        ax4.plot(com[0], com[1], 'go', markersize=8, label='COM')
        
        # Get payload position if available
        if hasattr(config.vehicle, 'payload_position'):
            locations = config.vehicle.payload_position
            if isinstance(locations, np.ndarray) and locations.size > 0:
                ax4.plot(locations[0], locations[1], 'r*', markersize=10, label='Payload')
                
                # Draw a line connecting COM and payload to visualize the offset
                ax4.plot([com[0], locations[0]], [com[1], locations[1]], 'k--', alpha=0.7)
                
                # Annotate the distance
                offset = (np.linalg.norm(locations[:2] - com[:2]) / config.vehicle.arm_length) * 100
                midpoint = (com[:2] + locations[:2]) 
                ax4.annotate(f"{offset:.2f} \% of arm length", xy=midpoint, xytext=(10, 10), 
                            textcoords="offset points", ha='center', va='bottom',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
        
        # Formatting
        ax4.set_xlabel('X [m]')
        ax4.set_ylabel('Y [m]')
        ax4.set_title('Payload and COM Location\n(Top View)')
        ax4.grid(True)
        ax4.legend(loc='upper right')
        
        # Set equal axis limits for better visualization
        max_range = max(
            abs(ax4.get_xlim()[1] - ax4.get_xlim()[0]),
            abs(ax4.get_ylim()[1] - ax4.get_ylim()[0])
        ) / 2.0
        mid_x = (ax4.get_xlim()[1] + ax4.get_xlim()[0]) / 2.0
        mid_y = (ax4.get_ylim()[1] + ax4.get_ylim()[0]) / 2.0
        ax4.set_xlim(mid_x - max_range, mid_x + max_range)
        ax4.set_ylim(mid_y - max_range, mid_y + max_range)
        
        return fig

    def _print_metrics(self, sim_results, controller_types):
        print("\nPerformance Metrics:")
        print("-" * 50)
        for result, ctrl_type in zip(sim_results, controller_types):
            pos_error, heading_error = compute_cost(result)
            print(f"\nController: {ctrl_type}")
            print(f"Average position error: {pos_error:.3f} m")
            print(f"Average heading error: {heading_error:.3f} deg")
        print("-" * 50)

    @dataclass
    class When2FailConfig:
        """Configuration for when2fail analysis visualization"""
        experiment_type: str
        controller_types: List[str]
        intensities: Dict[str, List[float]]
        success_rates: Dict[str, List[float]]
        pos_errors: Dict[str, List[float]]
        heading_errors: Dict[str, List[float]]

    def plot_when2fail_results(self, experiment_type, controller_types, intensities, 
                             success_rates, pos_errors, heading_errors):
        """Plot the results of the when2fail analysis with physical disturbance values"""
        # Create configuration object
        config = self.When2FailConfig(
            experiment_type=experiment_type,
            controller_types=controller_types,
            intensities=intensities,
            success_rates=success_rates,
            pos_errors=pos_errors,
            heading_errors=heading_errors
        )
        
        plt.figure(figsize=(3.5*3.5, 3.5))
        
        # Map intensities to actual physical quantities based on experiment type
        x_labels = {
            ExperimentType.WIND: (
                'Wind Speed (m/s)', 
                lambda i: i
            ),
            ExperimentType.FORCE: (
                'External Force Magnitude (N)', 
                lambda i: i
            ),
            ExperimentType.TORQUE: (
                'External Torque Magnitude (Nm)', 
                lambda i: i
            ),
            ExperimentType.UNCERTAINTY: (
                'Max $|c|$', 
                lambda i: 100 * i
            ),
            ExperimentType.ROTOR_EFFICIENCY: (
                'Max Rotor Efficiency Change (\%)', 
                lambda i: 100 * min(0.6, i)  # Special case as it's symmetric
            ),
            ExperimentType.PAYLOAD: (
                'Payload Mass Ratio (\%)', 
                lambda i: 100 * min(2.0, i)
            )
        }
        xlabel, transform_fn = x_labels.get(ExperimentType(config.experiment_type), ('Intensity Multiplier', lambda i: i))

        # Success Rate Plot
        plt.subplot(131)
        for ctrl in config.controller_types:
            x_values = [transform_fn(i) for i in config.intensities[ctrl]]
            plt.plot(x_values, config.success_rates[ctrl], marker='o', label=ctrl)
        plt.xlabel(xlabel)
        plt.ylabel('Success Rate (\%)')
        plt.grid(True)
        plt.legend()
        # Position Error Plot
        plt.subplot(132)
        for ctrl in config.controller_types:
            x_values = [transform_fn(i) for i in config.intensities[ctrl]]
            plt.plot(x_values, config.pos_errors[ctrl], marker='o', label=ctrl)
        plt.xlabel(xlabel)
        plt.ylabel('Average Position Error (m)')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        
        # Heading Error Plot
        plt.subplot(133)
        for ctrl in config.controller_types:
            x_values = [transform_fn(i) for i in config.intensities[ctrl]]
            plt.plot(x_values, config.heading_errors[ctrl], marker='o', label=ctrl)
        plt.xlabel(xlabel)
        plt.ylabel('Average Heading Error (deg)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / f'vis_when2fail_{config.experiment_type}_{"_".join(config.controller_types)}.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()        
        # Print summary with physical quantities
        print("\nWhen2Fail Analysis Summary:")
        print("-" * 50)
        print(f"Experiment type: {config.experiment_type}")
        for ctrl in config.controller_types:
            print(f"\n{ctrl}:")
            print(f"Final success rate: {config.success_rates[ctrl][-1]:.1f}%")
            print(f"Final position error: {config.pos_errors[ctrl][-1]:.2f} m")
            print(f"Final heading error: {config.heading_errors[ctrl][-1]:.2f} deg")
