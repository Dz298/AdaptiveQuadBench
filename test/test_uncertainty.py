import numpy as np
import pytest
from config.randomization_config import RandomizationConfig, ExperimentType, TrajectoryType, UncertantyType

def get_test_quad_params():
    """Returns a simple set of quadrotor parameters for testing"""
    return {
        'mass': 1.0,  # kg
        'arm_length': 0.2,  # m
        'Ixx': 0.01,  # kg*m^2
        'Iyy': 0.01,  # kg*m^2
        'Izz': 0.02,  # kg*m^2
        'k_eta': 1e-5,  # N/(rad/s)^2
        'k_m': 1e-6,   # N*m/(rad/s)^2
        'k_d': 0.1,    # drag coefficient
        'k_z': 0.2,    # vertical drag coefficient
        'k_flap': 0.3, # flapping coefficient
        'rotor_speed_min': 0,    # rad/s
        'rotor_speed_max': 800,  # rad/s
        'cd1x': 0.1,   # drag coefficient
        'cd1y': 0.1,
        'cd1z': 0.1,
        'cdz_h': 0.2,
        'c_Dx': 0.3,
        'c_Dy': 0.3,
        'c_Dz': 0.3,
        'rotor_pos': {
            'r1': np.array([0.1, -0.1, 0]),
            'r2': np.array([0.1, 0.1, 0]),
            'r3': np.array([-0.1, -0.1, 0]),
            'r4': np.array([-0.1, 0.1, 0])
        }
    }

class TestRandomization:
    @pytest.fixture
    def base_config(self):
        """Create a base configuration for testing"""
        return RandomizationConfig(
            num_trials=100,
            quad_params=get_test_quad_params(),
            seed=42,
            experiment_type=ExperimentType.UNCERTAINTY,
            trajectory_type=TrajectoryType.HOVER
        )

    def test_uniform_uncertainty(self, base_config):
        """Test uniform uncertainty randomization"""
        base_config.uncertainty_type = UncertantyType.UNIFORM
        base_config.controller_uncertainty_enabled = True
        base_config.uniform_model_uncertainty = 0.1  # 10% variation

        params_list = base_config.create_controller_params(get_test_quad_params())
        
        # Check we got the right number of parameter sets
        assert len(params_list) == base_config.num_trials

        # Test bounds for a few key parameters
        original_params = get_test_quad_params()
        for params in params_list:
            # Check mass bounds
            assert 0.9 * original_params['mass'] <= params['mass'] <= 1.1 * original_params['mass']
            
            # Check arm length bounds
            assert 0.9 * original_params['arm_length'] <= params['arm_length'] <= 1.1 * original_params['arm_length']
            
            # Check rotor positions scaled with arm length
            for key in params['rotor_pos']:
                print(params['rotor_pos'][key])
                print(original_params['rotor_pos'][key])
                assert np.allclose(
                    np.linalg.norm(params['rotor_pos'][key]),
                    np.linalg.norm(original_params['rotor_pos'][key] * params['arm_length'] / original_params['arm_length']),
                    rtol=1e-10
                )

    def test_scaled_uncertainty(self, base_config):
        """Test scaled uncertainty randomization"""
        base_config.uncertainty_type = UncertantyType.SCALED
        base_config.controller_uncertainty_enabled = True
        base_config.scaled_model_uncertainty = 0.1  # 10% variation
        base_config.scaled_model_uncertainty_noise = 0.0

        params_list = base_config.create_controller_params(get_test_quad_params())
        
        # Check we got the right number of parameter sets
        assert len(params_list) == base_config.num_trials

        original_params = get_test_quad_params()
        for params in params_list:
            # Get the scaling factor c from arm length
            c = params['arm_length']/original_params['arm_length'] - 1
            
            # Check mass scales with L^3
            expected_mass = original_params['mass'] * (1 + c)**3
            assert np.allclose(params['mass'], expected_mass)
            
            # Check inertia scales with L^5
            expected_Ixx = original_params['Ixx'] * (1 + c)**5
            assert np.allclose(params['Ixx'], expected_Ixx)
            
            # Check drag coefficients scale with L^2
            expected_cd = original_params['cd1x'] * (1 + c)**2
            assert np.allclose(params['cd1x'], expected_cd)
            
            # Check k_eta follows exponential formula
            expected_k_eta = max(1, 2.24e-8 * np.exp(32.78*params['arm_length']))
            assert np.allclose(params['k_eta'], expected_k_eta)
            
            # Check k_m maintains ratio with k_eta
            kappa = params['k_m'] / params['k_eta']
            expected_kappa = (1 + c) * original_params['k_m'] / original_params['k_eta']
            assert np.allclose(kappa, expected_kappa)

    def test_reproducibility(self, base_config):
        """Test that setting the same seed produces the same results"""
        base_config.seed = 42
        params_list1 = base_config.create_controller_params(get_test_quad_params())
        
        base_config.seed = 42
        params_list2 = base_config.create_controller_params(get_test_quad_params())
        
        # Check all parameters are identical
        for params1, params2 in zip(params_list1, params_list2):
            for key in params1:
                if isinstance(params1[key], dict):
                    for subkey in params1[key]:
                        assert np.allclose(params1[key][subkey], params2[key][subkey])
                else:
                    assert np.allclose(params1[key], params2[key])

if __name__ == "__main__":
    pytest.main([__file__]) 