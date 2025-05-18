import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.randomization_config import RandomizationConfig, ExperimentType
from quad_param.quadrotor import quad_params
from rotorpy.vehicles.multirotor import Multirotor
import numpy as np
def test_rotor_efficiency():
    # Create config with rotor efficiency enabled
    config = RandomizationConfig.from_experiment_type(
        ExperimentType.ROTOR_EFFICIENCY,
        num_trials=5,  # Small number for quick testing
        seed=42  # Fixed seed for reproducibility
    )
    
    # Generate vehicle parameters
    vehicle_params_list = config.create_vehicle_params(quad_params)
    
    # Print rotor efficiencies for each vehicle
    print("\nTesting Rotor Efficiency Randomization:")
    print("---------------------------------------")
    for i, params in enumerate(vehicle_params_list):
        print(f"Vehicle {i} rotor efficiencies:", params.get('rotor_efficiency', 'Not found!'))
        
    # Create vehicles and test thrust output
    print("\nTesting Thrust Output:")
    print("---------------------")
    base_vehicle = Multirotor(quad_params)
    control_input = {'cmd_motor_speeds': np.array([1000, 1000, 1000, 1000])}
    state = {'x': np.array([0, 0, 0]), 'v': np.array([0, 0, 0]), 'q': np.array([1, 0, 0, 0]), 
             'w': np.array([0, 0, 0]),'wind': np.array([0, 0, 0]),
             'rotor_speeds': np.array([1000, 1000, 1000, 1000]), 'ext_force': np.array([0, 0, 0]),
             'ext_torque': np.array([0, 0, 0])}
    for i, params in enumerate(vehicle_params_list):
        vehicle = Multirotor(params)
        
        # Test thrust with same rotor speeds # TODO: add print in Multirotor.step
        print(f"\nModified Vehicle {i}:")
        vehicle.step(state, control_input,0.01)
        print(f"\nBase Vehicle {i}:")
        base_vehicle.step(state, control_input,0.01)

if __name__ == "__main__":
    test_rotor_efficiency() 