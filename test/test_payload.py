import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.randomization_config import RandomizationConfig, ExperimentType
from quad_param.quadrotor import quad_params
import numpy as np
def test_payload():
    # Create config with payload enabled
    config = RandomizationConfig.from_experiment_type(
        ExperimentType.PAYLOAD,
        num_trials=5,  # Small number for quick testing
        quad_params=quad_params,
        seed=42  # Fixed seed for reproducibility
    )
    
    # Generate payload disturbances
    forces, torques = config.create_payload_disturbance()
    
    print("\nQuad Parameters:")
    print(f"Mass: {quad_params['mass']} kg")
    print(f"Arm Length: {quad_params['arm_length']} m")
    
    print("\nTesting Payload Disturbances:")
    print("-----------------------------")
    for i in range(len(forces)):
        print(f"\nPayload {i}:")
        print(f"Force (N): {forces[i]}")
        print(f"Torque (Nm): {torques[i]}")
        
        # Verify that force is always pointing downward
        assert forces[i][2] < 0, "Payload force should be downward"
        
        # Verify force and torque magnitudes are reasonable
        force_mag = np.linalg.norm(forces[i])
        torque_mag = np.linalg.norm(torques[i])
        
        print(f"Force magnitude: {force_mag:.3f} N")
        print(f"Torque magnitude: {torque_mag:.3f} Nm")
        
        # Check if magnitudes are within expected ranges
        max_expected_force = 0.5 * quad_params['mass'] * 9.81  # 50% of quad mass * gravity
        max_expected_torque = max_expected_force * 0.2 * quad_params['arm_length']  # max force * max offset ratio * arm length
        
        assert force_mag <= max_expected_force, f"Force magnitude {force_mag} exceeds maximum expected {max_expected_force}"
        assert torque_mag <= max_expected_torque, f"Torque magnitude {torque_mag} exceeds maximum expected {max_expected_torque}"
        
        # Print relative values
        print(f"Payload/Quad mass ratio: {force_mag/(9.81*quad_params['mass']):.1%}")
        print(f"Max torque/force ratio: {torque_mag/force_mag:.3f} m (should be ≤ {0.2*quad_params['arm_length']:.3f} m)")

if __name__ == "__main__":
    test_payload() 