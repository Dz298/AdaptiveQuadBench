from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import os
import csv
import time
from utils.parallel_data_collection import generate_data
from rotorpy.world import World
from config.simulation_config import SimulationConfig
class ResultsManager:
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / 'data'
        self.data_dir.mkdir(exist_ok=True)
        
    def generate_summary(self, config: SimulationConfig):
        """
        Generate a summary of simulation results using a configuration object.
        
        Args:
            config: SimulationConfig object containing all simulation parameters
        """
        controller_name = config.controller_type
        experiment_type = config.experiment_type
        
        output_csv_file = config.output_file if config.output_file else self.data_dir / f'summary_{controller_name}_{experiment_type}.csv'
        
        if output_csv_file.exists():
            output_csv_file.unlink()
            
        savepath = None
        if config.save_individual_trials:
            savepath = self.data_dir / f'trial_data_{controller_name}_{experiment_type}'
            self._handle_trial_directory(savepath)
            config.save_trial_path = savepath

        # Create headers
        with open(output_csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['traj_number', 'pos_tracking_error', 'heading_error'])

        # Generate data
        start_time = time.time()
        generate_data(output_csv_file, config)
        end_time = time.time()
        
        print(f"Time elapsed: {end_time-start_time:.2f} seconds, parallel: {config.parallel}")
        
        self._process_and_save_results(output_csv_file, controller_name, experiment_type)

    def _handle_trial_directory(self, savepath):
        if not savepath.exists():
            savepath.mkdir(parents=True)
        else:
            user_input = input(f"The directory {savepath} already exists. Do you want to remove the existing files? (y/n)")
            if user_input.lower() == 'y':
                for file in savepath.glob('*'):
                    file.unlink()
            elif user_input.lower() == 'n':
                raise Exception(f"Please delete or rename the files in the directory {savepath}")
            else:
                raise Exception("Invalid input. Please enter 'y' or 'n'")

    def _process_and_save_results(self, output_csv_file, controller_name, experiment_type):
        df = pd.read_csv(output_csv_file)
        success_rate = (df['pos_tracking_error'] < 5).sum() / len(df) * 100
        pos_tracking_error_success = df['pos_tracking_error'] < 5
        
        avg_pos_error = df['pos_tracking_error'][pos_tracking_error_success].mean()
        std_pos_error = df['pos_tracking_error'][pos_tracking_error_success].std()
        avg_heading_error = df['heading_error'][pos_tracking_error_success].mean()
        std_heading_error = df['heading_error'][pos_tracking_error_success].std()
        
        print("--------------------------------")
        print(f"Controller: {controller_name}")
        print("--------------------------------")
        print(f"Success rate: {success_rate:.2f}%")
        print(f"Average pos_tracking_error: {avg_pos_error:.2f} m")
        print(f"Std pos_tracking_error: {std_pos_error:.2f} m")
        print(f"Average heading_error: {avg_heading_error:.2f} deg")
        print(f"Std heading_error: {std_heading_error:.2f} deg")
        print("--------------------------------")

        self._update_stats_csv(
            controller_name,
            experiment_type,
            success_rate,
            avg_pos_error,
            avg_heading_error,
            std_pos_error,
            std_heading_error
        )

    def _update_stats_csv(self, controller_name, experiment_type, success_rate, 
                         avg_pos_error, avg_heading_error, std_pos_error, std_heading_error):
        stats_file = self.data_dir / 'controller_stats.csv'
        
        new_stats = pd.DataFrame({
            'controller': [controller_name],
            'experiment': [experiment_type],
            'success_rate': [success_rate],
            'avg_position_error': [avg_pos_error],
            'std_position_error': [std_pos_error],
            'avg_heading_error': [avg_heading_error],
            'std_heading_error': [std_heading_error],
            'last_updated': [pd.Timestamp.now()]
        })
        
        try:
            if stats_file.exists():
                stats_df = pd.read_csv(stats_file)
                stats_df = stats_df[~((stats_df['controller'] == controller_name) & 
                                    (stats_df['experiment'] == experiment_type))]
                stats_df = pd.concat([stats_df, new_stats], ignore_index=True)
            else:
                stats_df = new_stats

            stats_df.to_csv(stats_file, index=False)
            print(f"\nUpdated statistics in {stats_file}")
            print(stats_df.to_string(index=False))

        except Exception as exp:
            print(f"Error updating stats file: {exp}")