import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.randomization_config import RandomizationConfig, ExperimentType
from quad_param.quadrotor import quad_params
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.world import World
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.environments import Environment
from controller.geometric_control import GeoControl
from rotorpy.wind.dryden_winds import DrydenGust
import numpy as np

def test_random_disturbance():
    # Create a simple world
    world = World.empty(((-5, 5), (-5, 5), (0, 5)))
    
    # Create a vehicle
    vehicle = Multirotor(quad_params)
    
    # Create a simple hover trajectory
    trajectory = HoverTraj()
    
    # Create a controller
    controller = GeoControl(quad_params)
    
    # Create a wind profile (no wind for this test)
    wind_profile = DrydenGust(dt=0.01, avg_wind=np.zeros(3), sig_wind=np.zeros(3))
    
    # Set up constant disturbances
    ext_force = np.array([1.0, 0.0, 0.0])  # 1N force in x direction
    ext_torque = np.array([0.0, 0.1, 0.0])  # 0.1Nm torque around y axis
    
    # Initial state
    initial_state = {
        'x': np.array([0, 0, 0]),
        'v': np.zeros(3),
        'q': np.array([0, 0, 0, 1]),
        'w': np.zeros(3),
        'rotor_speeds': np.array([0, 0, 0, 0]),
        'wind': np.zeros(3),
        'ext_force': np.zeros(3),
        'ext_torque': np.zeros(3)
    }
    
    # Create simulator instance
    sim = Environment(
        vehicle=vehicle,
        controller=controller,
        trajectory=trajectory,
        wind_profile=wind_profile,
        ext_force=ext_force,
        ext_torque=ext_torque
    )
    
    # Run simulation with random disturbances enabled
    print("\nTesting Random Disturbance:")
    print("-------------------------")
    result = sim.run(
        t_final=5.0,
    )
    # Check if disturbances were toggled
    state_history = result['state']  # Get state history from result
    force_history = state_history['ext_force']
    torque_history = state_history['ext_torque']
    # Count number of disturbance changes
    force_changes = np.sum(np.diff(force_history, axis=0) != 0)
    torque_changes = np.sum(np.diff(torque_history, axis=0) != 0)
    
    print(f"\nNumber of force toggles: {force_changes}")
    print(f"Number of torque toggles: {torque_changes}")
    
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_random_disturbance()
