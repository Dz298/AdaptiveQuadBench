from typing import List, Optional, Callable
import numpy as np
import time
from pathlib import Path
from quad_param.quadrotor import quad_params
from rotorpy.world import World
from rotorpy.vehicles.multirotor import Multirotor
from config.randomization_config import RandomizationConfig
from .config_manager import ExperimentConfig
from .results_manager import ResultsManager
from .visualizer import ExperimentVisualizer
import pandas as pd
import os
import matplotlib.pyplot as plt
from rotorpy.environments import Environment
from utils.delay_analysis import compute_delay_margin, plot_delay_margin_results, visualize_delay_response, generate_random_trajectories, plot_multi_trajectory_results
from rotorpy.trajectories.circular_traj import CircularTraj
from dataclasses import dataclass, field
from typing import Any, Dict
from config.simulation_config import SimulationConfig



class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, controller_factory: Callable):
        self.config = config
        self.controller_factory = controller_factory
        self.results_manager = ResultsManager()
        self.visualizer = ExperimentVisualizer()
        
    def run(self):
        if self.config.when2fail:
            self._run_when2fail()
        elif self.config.delay_margin:
            return self.run_delay_margin_experiment()
        elif self.config.visualize:
            self._run_visualization()
        else:
            self._run_normal_experiments()
            
    def _run_normal_experiments(self):
        rand_config = RandomizationConfig.from_experiment_type(
            self.config.experiment_type,
            self.config.num_trials,
            quad_params,
            self.config.seed,
            trajectory_type=self.config.trajectory_type
        )
        
        components = self._generate_components(rand_config)
        
        for controller_type in self.config.controller_types:
            print(f"\nRunning experiments for controller: {controller_type}")
            print("=" * 50)
            
            controllers = [
                self.controller_factory(controller_type, params) 
                for params in components['controller_params']
            ]
            
            use_parallel = (not self.config.use_serial) and controller_type != 'xadap'
            
            # Create a world
            world_size = 10
            world = World.empty([-world_size/2, world_size/2, -world_size/2, world_size/2, -world_size/2, world_size/2])
            
            # Create simulation config
            sim_config = SimulationConfig(
                world=world,
                vehicles=components['vehicles'],
                controllers=controllers,
                wind_profiles=components['wind_profiles'],
                trajectories=components['trajectories'],
                num_simulations=self.config.num_trials,
                ext_force=components['ext_force'],
                ext_torque=components['ext_torque'],
                parallel=use_parallel,
                save_individual_trials=self.config.save_trials,
                experiment_type=self.config.experiment_type,
                controller_type=controller_type
            )
            
            # Generate summary with the config object
            self.results_manager.generate_summary(sim_config)

    def _generate_components(self, config):
        trajectories = config.create_trajectories()
        wind_profiles = config.create_wind_profiles()
        ext_force, ext_torque = config.create_ext_force_and_torque()
        payload_masses, payload_positions = config.create_payload_disturbance()
        vehicle_params_list = config.create_vehicle_params(quad_params)
        vehicles = [Multirotor(params) for params in vehicle_params_list]
        controller_params_list = config.create_controller_params(quad_params)

        # Apply payload disturbance to vehicles
        self._apply_payload_to_vehicles(vehicles, payload_masses, payload_positions)

        return {
            'trajectories': trajectories,
            'wind_profiles': wind_profiles,
            'ext_force': ext_force,
            'ext_torque': ext_torque,
            'vehicles': vehicles,
            'controller_params': controller_params_list,
            'vehicle_params': vehicle_params_list
        }

    def _apply_payload_to_vehicles(self, vehicles, payload_masses, payload_positions):
        """
        Update payload parameters to vehicles.
        
        Args:
            vehicles: List of Multirotor instances
            payload_masses: Array of payload masses
            payload_positions: Array of payload positions
        """
        if payload_masses is None:
            return
        
        for i, (vehicle, payload_mass, payload_position) in enumerate(zip(vehicles, payload_masses, payload_positions)):
            # Update vehicle physical properties (mass, COM, inertia)
            vehicle.update_payload(payload_mass, payload_position)
    def _run_visualization(self):
        print("Visualization mode: Running single trial with multiple controllers")
        
        rand_config = RandomizationConfig.from_experiment_type(
            self.config.experiment_type,
            1,  # Single trial for visualization
            quad_params,
            self.config.seed,
            trajectory_type=self.config.trajectory_type
        )
        
        components = self._generate_components(rand_config)
        controllers = [
            self.controller_factory(ctrl_type, components['controller_params'][0]) 
            for ctrl_type in self.config.controller_types
        ]
        
        # Run simulations and collect results
        sim_results = []
        for vehicle, controller in zip([components['vehicles'][0]] * len(controllers), controllers):
            controller.update_trajectory(components['trajectories'][0])
            
            sim_instance = Environment(
                vehicle=vehicle, 
                controller=controller,
                wind_profile=components['wind_profiles'][0] if components['wind_profiles'][0] is not None else None,
                trajectory=components['trajectories'][0],
                sim_rate=100,
                ext_force=components['ext_force'][0] if components['ext_force'] is not None else None,
                ext_torque=components['ext_torque'][0] if components['ext_torque'] is not None else None
            )

            x0 = {
                'x': np.array([0, 0, 0]),
                'v': np.zeros(3,),
                'q': np.array([0, 0, 0, 1]),
                'w': np.zeros(3,),
                'wind': np.array([0,0,0]),
                'rotor_speeds': np.array([0,0,0,0])
            }

            sim_result = sim_instance.run(
                t_final=5,
                use_mocap=False,
                terminate=False,
                plot=False,
                animate_bool=False,
                verbose=False
            )
            sim_results.append(sim_result)
        
        # Visualize results
        self.visualizer.visualize_trials(
            experiment_type=self.config.experiment_type,
            sim_results=sim_results,
            controller_types=self.config.controller_types,
            controller_param=components['controller_params'][0],
            vehicle_params=components['vehicle_params'][0],
            vehicle=components['vehicles'][0]
        )

    def _run_when2fail(self):
        """Run experiments with increasing disturbance intensity until failure or max intensity"""

        # Initialize tracking variables
        intensities = {ctrl: [] for ctrl in self.config.controller_types}
        success_rates = {ctrl: [] for ctrl in self.config.controller_types}
        pos_errors = {ctrl: [] for ctrl in self.config.controller_types}
        heading_errors = {ctrl: [] for ctrl in self.config.controller_types}
        
        print(f"\nRunning when2fail analysis for experiment type: {self.config.experiment_type}")
        
        # Create base config and get constant components
        rand_config = RandomizationConfig.from_experiment_type(
            self.config.experiment_type,
            self.config.num_trials,
            quad_params,
            self.config.seed,
            trajectory_type=self.config.trajectory_type
        )
        base_components = rand_config.create_base_components()
        
        # Start with original intensity
        intensity = 0
        active_controllers = self.config.controller_types.copy()
        
        while intensity <= self.config.max_intensity and active_controllers:
            print(f"\nTesting intensity multiplier: {intensity:.1f}")
            print(f"Active controllers: {active_controllers}")
            
            # Scale ranges for current intensity
            rand_config.scale_ranges_with_intensity(intensity)
            
            # Generate intensity-dependent components
            varied_components = rand_config.create_varied_components()
            
            # Create vehicles from parameters
            vehicles = [Multirotor(params) for params in 
                       (varied_components.get('vehicle_params') or base_components['vehicle_params'])]
            
            if varied_components['payload_masses'] is not None:
                self._apply_payload_to_vehicles(
                    vehicles, 
                    varied_components['payload_masses'], 
                    varied_components['payload_positions'], 
                )
            
            # Create controllers using appropriate parameters
            controller_params = varied_components.get('controller_params') or base_components['controller_params']
            
            # Test each active controller
            controllers_to_remove = []
            for controller_type in active_controllers:
                print(f"Testing {controller_type}...")
                
                controllers = [
                    self.controller_factory(controller_type, params) 
                    for params in controller_params
                ]
                
                # Create temporary CSV for this run
                temp_csv = self.results_manager.data_dir / f'temp_{controller_type}_{intensity:.1f}.csv'
                
                # Create a world
                world_size = 10
                world = World.empty([-world_size/2, world_size/2, -world_size/2, world_size/2, -world_size/2, world_size/2])
                use_parallel = (not self.config.use_serial) and controller_type != 'xadap'
                # Create simulation config
                sim_config = SimulationConfig(
                    world=world,
                    vehicles=vehicles,
                    controllers=controllers,
                    wind_profiles=varied_components['wind_profiles'],
                    trajectories=base_components['trajectories'],
                    num_simulations=self.config.num_trials,
                    ext_force=varied_components['ext_force'],
                    ext_torque=varied_components['ext_torque'],
                    parallel=use_parallel,
                    save_individual_trials=self.config.save_trials,
                    experiment_type=self.config.experiment_type,
                    controller_type=controller_type,
                    output_file=temp_csv
                )
                
                # Generate summary with the config object
                self.results_manager.generate_summary(sim_config)
                
                # Read results
                df = pd.read_csv(temp_csv)
                success_rate = (df['pos_tracking_error'] < 5).mean() * 100
                avg_pos_error = df['pos_tracking_error'].mean()
                avg_heading_error = df['heading_error'].mean()
                
                # Store results
                intensities[controller_type].append(intensity)
                success_rates[controller_type].append(success_rate)
                pos_errors[controller_type].append(avg_pos_error)
                heading_errors[controller_type].append(avg_heading_error)
                
                # Check if controller has failed (0% success rate)
                if success_rate == 0:
                    controllers_to_remove.append(controller_type)
                    print(f"{controller_type} failed at intensity {intensity:.1f}")
                
                # Clean up temporary file
                temp_csv.unlink()
            
            # Remove failed controllers
            for ctrl in controllers_to_remove:
                active_controllers.remove(ctrl)
            
            # Increment intensity
            intensity += self.config.intensity_step
        
        # Visualize results with rand_config
        self.visualizer.plot_when2fail_results(
            experiment_type=self.config.experiment_type,
            controller_types=self.config.controller_types,
            intensities=intensities,
            success_rates=success_rates,
            pos_errors=pos_errors,
            heading_errors=heading_errors,
        )

    def run_delay_margin_experiment(self):
        """Run experiments to determine delay margin across multiple trajectories."""
        print("\nRunning Delay Margin Experiment...")
        
        # Create results directory if it doesn't exist
        results_dir = self.results_manager.data_dir / 'delay_margin'
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate multiple trajectories
        num_trajectories = 5  # Total number of trajectories to test
        trajectories, trajectory_types = generate_random_trajectories(num_trajectories)
        
        # Store results for all controllers and trajectories
        all_results = {}
        
        for controller_type in self.config.controller_types:
            print(f"\nTesting delay margin for {controller_type}...")
            
            controller_results = {}
            controller_delay_margins = []
            
            # Create controller-specific directory
            controller_dir = results_dir / controller_type
            os.makedirs(controller_dir, exist_ok=True)
            
            # Test each trajectory
            for traj_idx, (trajectory, traj_name) in enumerate(zip(trajectories, trajectory_types)):
                print(f"  Testing trajectory: {traj_name}")
                
                # Compute delay margin for this trajectory
                margin, results = compute_delay_margin(
                    controller_factory=self.controller_factory,
                    controller_type=controller_type,
                    vehicle_params=quad_params,
                    trajectory=trajectory,
                    initial_delay=0.0,
                    max_delay=0.5,
                    delay_step=0.01,
                    test_duration=5.0,
                    position_threshold=1.0
                )
                
                controller_delay_margins.append(margin)
                controller_results[traj_name] = {
                    'delay_margin': margin,
                    'results': results
                }
                
                # Visualize individual trajectory response
                fig_path = controller_dir / f'delay_response_{traj_name}.png'
                visualize_delay_response(controller_type, results, save_path=fig_path)
            
            # Store results for this controller
            all_results[controller_type] = {
                'delay_margins': controller_delay_margins,
                'trajectory_types': trajectory_types,
                'detailed_results': controller_results
            }
            
            # Plot delay margins across trajectories for this controller
            fig_path = controller_dir / 'trajectory_comparison.png'
            plot_multi_trajectory_results(
                controller_type, 
                trajectory_types, 
                controller_delay_margins,
                save_path=fig_path
            )
            
            # Calculate statistics
            avg_margin = np.mean(controller_delay_margins)
            std_margin = np.std(controller_delay_margins)
            min_margin = np.min(controller_delay_margins)
            max_margin = np.max(controller_delay_margins)
            
            print(f"  Delay margin statistics:")
            print(f"    Average: {avg_margin:.3f} ± {std_margin:.3f} seconds")
            print(f"    Range: [{min_margin:.3f}, {max_margin:.3f}] seconds")
            
            # Save detailed results to CSV
            results_data = []
            for traj_name, margin in zip(trajectory_types, controller_delay_margins):
                results_data.append({
                    'trajectory': traj_name,
                    'delay_margin': margin
                })
            
            results_df = pd.DataFrame(results_data)
            csv_path = controller_dir / 'delay_margins.csv'
            results_df.to_csv(csv_path, index=False)
        
        # Compare controllers using average delay margins
        avg_margins = [np.mean(all_results[ctrl]['delay_margins']) for ctrl in self.config.controller_types]
        
        # Plot comparison of average delay margins across controllers
        fig_path = results_dir / 'controller_comparison.png'
        plot_delay_margin_results(
            self.config.controller_types,
            avg_margins,
            detailed_results=all_results,
            save_path=fig_path
        )
        
        # Save summary results
        summary_data = []
        for ctrl_type in self.config.controller_types:
            margins = all_results[ctrl_type]['delay_margins']
            summary_data.append({
                'controller': ctrl_type,
                'avg_delay_margin': np.mean(margins),
                'std_delay_margin': np.std(margins),
                'min_delay_margin': np.min(margins),
                'max_delay_margin': np.max(margins)
            })
        
        summary_df = pd.DataFrame(summary_data)
        csv_path = results_dir / 'summary.csv'
        summary_df.to_csv(csv_path, index=False)
        
        print(f"\nDelay margin analysis complete. Results saved to {results_dir}")
        
        return all_results
