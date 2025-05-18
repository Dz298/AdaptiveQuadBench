# Import your controller here
from controller.geometric_control import GeoControl
from controller.geometric_adaptive_controller import GeometricAdaptiveController
from controller.geometric_control_l1 import L1_GeoControl
from controller.indi_adaptive_controller import INDIAdaptiveController
from controller.quadrotor_control_mpc import ModelPredictiveControl
from controller.quadrotor_control_mpc_l1 import L1_ModelPredictiveControl
from controller.Xadap_NN_control import Xadap_NN_control
from rotorpy.controllers.quadrotor_control import SE3Control

# Import core components
from experiments.config_manager import ExperimentConfig, parse_experiment_args
from experiments.experiment_runner import ExperimentRunner
from experiments.results_manager import ResultsManager
from experiments.visualizer import ExperimentVisualizer

def switch_controller(controller_type, quad_params):
    if controller_type == 'geo':
        return GeoControl(quad_params)
    elif controller_type == 'geo-a':
        return GeometricAdaptiveController(quad_params)
    elif controller_type == 'l1geo':
        return L1_GeoControl(quad_params)
    elif controller_type == 'indi-a':
        return INDIAdaptiveController(quad_params)
    elif controller_type == 'l1mpc':
        return L1_ModelPredictiveControl(quad_params)
    elif controller_type == 'mpc':
        return ModelPredictiveControl(quad_params)
    elif controller_type == 'xadap':
        return Xadap_NN_control(quad_params)
    else:
        raise ValueError(f"Controller type {controller_type} not supported yet.")

def main():
    config = parse_experiment_args()
    runner = ExperimentRunner(config, switch_controller)
    runner.run()

if __name__ == '__main__':
    main()