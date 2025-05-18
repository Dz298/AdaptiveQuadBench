from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any
from enum import Enum, auto
import numpy as np
from rotorpy.trajectories.random_motion_prim_traj import RapidTrajectory
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.circular_traj import CircularTraj
from rotorpy.wind.dryden_winds import DrydenGust
from copy import deepcopy


class ExperimentType(Enum):
    """Enum for different experiment types"""
    NO = 'no'  # No randomization or disturbances
    WIND = 'wind'
    UNCERTAINTY = 'uncertainty'
    FORCE = 'force'
    TORQUE = 'torque'
    ROTOR_EFFICIENCY = 'rotoreff'
    PAYLOAD = 'payload'

    @classmethod
    def from_string(cls, value: str) -> 'ExperimentType':
        """Convert string to ExperimentType, case-insensitive"""
        try:
            return cls(value.lower())
        except ValueError:
            valid_types = [e.value for e in cls]
            raise ValueError(f"Invalid experiment type: {value}. Must be one of {valid_types}")
    
    def __str__(self):
        return self.value

class TrajectoryType(Enum):
    """Enum for different trajectory types"""
    RANDOM = 'random'  # Current random motion primitive
    HOVER = 'hover'    # Hover at a point
    CIRCLE = 'circle'  # Circular trajectory
    
    @classmethod
    def from_string(cls, value: str) -> 'TrajectoryType':
        """Convert string to TrajectoryType, case-insensitive"""
        try:
            return cls(value.lower())
        except ValueError:
            valid_types = [e.value for e in cls]
            raise ValueError(f"Invalid trajectory type: {value}. Must be one of {valid_types}")
    
    def __str__(self):
        return self.value
    
class UncertantyType(Enum):
    UNIFORM = 'uniform'
    SCALED = 'scaled'

@dataclass
class RandomizationConfig:
    """Configuration class for all randomization parameters"""
    num_trials: int
    quad_params: Dict[str, Any]
    seed: Optional[int] = None
    experiment_type: ExperimentType = ExperimentType.NO
    uncertainty_type: UncertantyType = UncertantyType.UNIFORM
    trajectory_type: TrajectoryType = TrajectoryType.RANDOM
    
    # Current ranges (with intensity scaling)
    wind_speed_range: tuple = (0, 3)
    force_range: tuple = (0, 1)
    torque_range: tuple = (0, 0.1)
    uniform_model_uncertainty: float = 0.1
    scaled_model_uncertainty: float = 0.1
    scaled_model_uncertainty_noise: float = 0.2
    rotor_efficiency_range: tuple = (-0.3, 0.3)
    payload_mass_ratio_range: tuple = (0.0, 0.5)
    
    # Fixed ranges (not affected by intensity)
    wind_sigma_range: tuple = (30, 60)
    payload_offset_ratio_range: tuple = (-0.2,0.2)
    traj_pos_range: tuple = (-2, 2)
    traj_vel_range: tuple = (-2, 2)
    traj_acc_range: tuple = (-2, 2)
    traj_time: float = 5.0

    # Enable flags
    wind_enabled: bool = False
    ext_force_enabled: bool = False
    ext_torque_enabled: bool = False
    controller_uncertainty_enabled: bool = False
    rotor_efficiency_enabled: bool = False
    payload_enabled: bool = False

    def __post_init__(self):
        """Initialize random seed if provided"""
        if self.seed is not None:
            np.random.seed(self.seed)

    @classmethod
    def from_experiment_type(cls, experiment_type: Union[str, ExperimentType], 
                           num_trials: int, 
                           quad_params: Dict[str, Any], 
                           seed: Optional[int] = None,
                           trajectory_type: Union[str, TrajectoryType] = TrajectoryType.RANDOM) -> 'RandomizationConfig':
        """Factory method to create config"""
        if isinstance(experiment_type, str):
            experiment_type = ExperimentType.from_string(experiment_type)
        if isinstance(trajectory_type, str):
            trajectory_type = TrajectoryType.from_string(trajectory_type)
            
        # Create base configuration
        base_config = {
            'num_trials': num_trials,
            'quad_params': quad_params,
            'seed': seed,
            'wind_enabled': False,
            'controller_uncertainty_enabled': False,
            'ext_force_enabled': False,
            'ext_torque_enabled': False,
            'rotor_efficiency_enabled': False,
            'payload_enabled': False,
            'experiment_type': experiment_type,
            'trajectory_type': trajectory_type
        }
        
        # Update enabled flags based on experiment type
        configs = {
            ExperimentType.NO: {},
            ExperimentType.WIND: {'wind_enabled': True},
            ExperimentType.UNCERTAINTY: {'controller_uncertainty_enabled': True},
            ExperimentType.FORCE: {'ext_force_enabled': True},
            ExperimentType.TORQUE: {'ext_torque_enabled': True},
            ExperimentType.ROTOR_EFFICIENCY: {'rotor_efficiency_enabled': True},
            ExperimentType.PAYLOAD: {'payload_enabled': True},
         
        }
        
        base_config.update(configs[experiment_type])
        
        # Create config instance
        config = cls(**base_config)
        
        return config
    

    def scale_ranges_with_intensity(self, intensity: float):
        """Scale the ranges based on intensity and experiment type"""
        if intensity < 0:
            raise ValueError("Intensity must be non-negative")

        if self.experiment_type == ExperimentType.NO:
            raise ValueError("No scaling for no experiment type")
            
        if self.experiment_type == ExperimentType.WIND:
            # Only scale wind speed
            max_wind = intensity
            self.wind_speed_range = (max_wind, max_wind)
            
        elif self.experiment_type == ExperimentType.FORCE:
            # Only scale force
            max_force = intensity
            self.force_range = (max_force, max_force)
            
        elif self.experiment_type == ExperimentType.TORQUE:
            # Only scale torque
            max_torque = intensity
            self.torque_range = (max_torque, max_torque)
            
        elif self.experiment_type == ExperimentType.UNCERTAINTY:
            # Only do scaled uncertainties 
            self.uncertainty_type = UncertantyType.SCALED
            self.scaled_model_uncertainty = intensity
            
        elif self.experiment_type == ExperimentType.ROTOR_EFFICIENCY:
            # Only scale rotor efficiency
            min_efficiency = max(-0.6, -intensity)
            max_efficiency = min(0.6, intensity)
            self.rotor_efficiency_range = (min_efficiency, max_efficiency)
            
        elif self.experiment_type == ExperimentType.PAYLOAD:
            # Only scale payload mass
            max_payload = min(2.0, intensity)
            self.payload_mass_ratio_range = (max_payload, max_payload)
            
    def create_base_components(self) -> Dict[str, Any]:
        """Generate components that should stay constant across intensity variations"""
        base_components = {
            'trajectories': self.create_trajectories(),
        }

        # Add components based on experiment type
        if self.experiment_type in [ExperimentType.NO, ExperimentType.WIND, ExperimentType.FORCE, ExperimentType.TORQUE, ExperimentType.PAYLOAD]:
            # For these types, vehicle and controller params stay constant
            base_components['vehicle_params'] = self.create_vehicle_params(self.quad_params)
            base_components['controller_params'] = self.create_controller_params(self.quad_params)
        
        return base_components

    def create_varied_components(self) -> Dict[str, Any]:
        """Generate components that vary with intensity based on experiment type"""
        varied_components = {}

        varied_components['wind_profiles'] = self.create_wind_profiles()
        varied_components['ext_force'], varied_components['ext_torque'] = self.create_ext_force_and_torque()
        varied_components['payload_masses'], varied_components['payload_positions'] = self.create_payload_disturbance()
        if self.experiment_type == ExperimentType.ROTOR_EFFICIENCY or self.experiment_type == ExperimentType.UNCERTAINTY:
            varied_components['vehicle_params'] = self.create_vehicle_params(self.quad_params)
            varied_components['controller_params'] = self.create_controller_params(self.quad_params)
            
        return varied_components


    def create_trajectories(self) -> List[Union[RapidTrajectory, HoverTraj, CircularTraj]]:
        """Generate trajectories based on specified type"""
        trajectories = []
        for _ in range(self.num_trials):
            if self.trajectory_type == TrajectoryType.RANDOM:
                # Current random motion primitive
                pos0 = np.array([0, 0, 0])
                vel0 = np.array([0, 0, 0])
                acc0 = np.array([0, 0, 0])
                gravity = np.array([0, 0, -9.81])
                
                trajectory = RapidTrajectory(pos0, vel0, acc0, gravity)
                posf = np.random.uniform(*self.traj_pos_range, size=3)
                velf = np.random.uniform(*self.traj_vel_range, size=3)
                accf = np.random.uniform(*self.traj_acc_range, size=3)
                
                trajectory.set_goal_position(posf)
                trajectory.set_goal_velocity(velf)
                trajectory.set_goal_acceleration(accf)
                trajectory.generate(self.traj_time)
                
            elif self.trajectory_type == TrajectoryType.HOVER:
                # Hover at random point
                trajectory = HoverTraj()
                
            elif self.trajectory_type == TrajectoryType.CIRCLE:
                # Create circle with center at (-radius, 0, 0) so it starts at (0,0,0)
                radius = 2
                trajectory = CircularTraj(center=np.array([-radius, 0, 0]), radius=radius)
            else:
                raise ValueError(f"Invalid trajectory type: {self.trajectory_type}")
            trajectories.append(trajectory)
            
        return trajectories
    
    def create_wind_profiles(self) -> List[Optional[DrydenGust]]:
        """Generate randomized wind profiles"""
        if not self.wind_enabled:
            return [None] * self.num_trials
            
        wind_profiles = []
        for _ in range(self.num_trials):
            spd = np.random.uniform(*self.wind_speed_range)
            dir = np.random.uniform(-1,1, size=3)
            dir = dir / np.linalg.norm(dir)
            w = spd * dir
            s = np.random.uniform(*self.wind_sigma_range, size=3)
            
            wind_profile = DrydenGust(
                dt=1/100,
                avg_wind=w,
                sig_wind=s
            )
            wind_profiles.append(wind_profile)
        return wind_profiles
    
    def create_vehicle_params(self, base_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate vehicle parameters with optional rotor efficiency variations"""
        vehicle_params_list = []
        for _ in range(self.num_trials):
            params = deepcopy(base_params)
            
            # Apply rotor efficiency to vehicle parameters
            if self.rotor_efficiency_enabled:
                params['rotor_efficiency'] += np.random.uniform(*self.rotor_efficiency_range, size=4)
            
            vehicle_params_list.append(params)
        return vehicle_params_list

    def create_payload_disturbance(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Generate payload masses and positions for disturbance testing.
        
        This method only generates the physical parameters of the payload (mass and position).
        The actual application of the payload to vehicles and calculation of resulting forces
        and torques should be handled separately.
        
        Returns:
            tuple: (payload_masses, payload_positions) or (None, None) if payload is disabled
        """
        if not self.payload_enabled:
            return None, None
            
        payload_masses = []
        payload_positions = []
        
        # Use stored quad_params
        quad_mass = self.quad_params.get('mass', 1.0)
        quad_arm_length = self.quad_params.get('arm_length', 0.3)
        
        for _ in range(self.num_trials):
            mass_ratio = np.random.uniform(*self.payload_mass_ratio_range)
            mass = mass_ratio * quad_mass
            
            offset_ratio = np.random.uniform(*self.payload_offset_ratio_range, size=3)
            offset = offset_ratio * quad_arm_length
            payload_masses.append(mass)
            payload_positions.append(offset)
            
        return np.array(payload_masses), np.array(payload_positions)
    
    def create_ext_force_and_torque(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Generate randomized external forces and torques"""
        ext_force = np.zeros((self.num_trials, 3))
        ext_torque = np.zeros((self.num_trials, 3))
        if self.ext_force_enabled:
            ext_force += np.random.uniform(*self.force_range, size=(self.num_trials, 3))
        if self.ext_torque_enabled:
            ext_torque += np.random.uniform(*self.torque_range, size=(self.num_trials, 3))
        return ext_force, ext_torque

    
    def create_controller_params(self, base_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate controller parameters with uncertainty"""
        controller_params_list = []
        for _ in range(self.num_trials):
            params = deepcopy(base_params)
            if self.controller_uncertainty_enabled:
                if self.uncertainty_type == UncertantyType.UNIFORM:
                    original_arm_length = base_params['arm_length']
                    for key, value in params.items():
                        if key in ['mass', 'Ixx', 'Iyy', 'Izz', 'arm_length', 'c_Dx', 'c_Dy', 'c_Dz', 
                                'rotor_speed_min', 'rotor_speed_max', 'k_eta', 'k_m', 'k_d', 'k_z',
                                'k_flap', 'cd1_x', 'cd1_y', 'cd1_z', 'cdz_h']:
                            params[key] *= (1 + np.random.uniform(-self.uniform_model_uncertainty, self.uniform_model_uncertainty))

                    params['rotor_pos'] = {rotor_key: rotor_value * params['arm_length'] / base_params['arm_length'] 
                                         for rotor_key, rotor_value in params['rotor_pos'].items()}
                else: # Scaled Uncertainty
                    # scaling constant 
                    c = max(-1+0.001,np.random.uniform(-self.scaled_model_uncertainty, self.scaled_model_uncertainty))
                    # Linear scaling componnets
                    params['arm_length'] = (1 + c) * base_params['arm_length']
                    kappa = (1+c) * base_params['k_m'] / base_params['k_eta']
                    params['k_d'] = (1+c) * base_params['k_d']
                    params['k_z'] = (1+c) * base_params['k_z']
                    params['k_flap'] = (1+c) * base_params['k_flap']

                    # We are not scaling the rotor speed max since it is for clamping 
                    # params['rotor_speed_max'] = (1+c) * base_params['rotor_speed_max']
                    
                    # Calculate scaling factors
                    l_to_m = (1 + c)**3  # mass scales with L^3
                    I_to_m = (1 + c)**5  # inertia scales with L^5 
                    cd_to_m = (1 + c)**2  # drag coefficients scale with L^2
                    
                    # k_eta calculation using exponential formula
                    # fitted with (crazyfile, humming bird, Agilicious, and 2 lab custom quadrotors)
                    params['k_eta'] = min(1, 5.385e-8 * np.exp(4.73*c))

                    # Apply scaling to other parameters
                    params['mass'] = base_params['mass'] * l_to_m
                    params['Ixx'] = base_params['Ixx'] * I_to_m
                    params['Iyy'] = base_params['Iyy'] * I_to_m
                    params['Izz'] = base_params['Izz'] * I_to_m
                    params['cd1x'] = base_params['cd1x'] * cd_to_m
                    params['cd1y'] = base_params['cd1y'] * cd_to_m
                    params['cd1z'] = base_params['cd1z'] * cd_to_m
                    params['cdz_h'] = base_params['cdz_h'] * cd_to_m
                    params['c_Dx'] = base_params['c_Dx'] * cd_to_m
                    params['c_Dy'] = base_params['c_Dy'] * cd_to_m
                    params['c_Dz'] = base_params['c_Dz'] * cd_to_m
                    params['k_m'] = kappa * params['k_eta']

                    # Apply uniform ±20% noise to all scaled parameters
                    noise_params = [
                        'arm_length', 'k_d', 'k_z', 'k_flap', 'rotor_speed_max',
                        'mass', 'Ixx', 'Iyy', 'Izz',
                        'cd1x', 'cd1y', 'cd1z', 'cdz_h',
                        'c_Dx', 'c_Dy', 'c_Dz',
                        'k_eta', 'k_m'
                    ]
                    
                    for key in noise_params:
                        noise = np.random.uniform(-self.scaled_model_uncertainty_noise, self.scaled_model_uncertainty_noise)  # ±20% uniform noise
                        params[key] *= (1 + noise)

                    # Update rotor positions
                    params['rotor_pos'] = {rotor_key: rotor_value * params['arm_length'] / base_params['arm_length'] 
                                         for rotor_key, rotor_value in params['rotor_pos'].items()}

            controller_params_list.append(params)
        
        return controller_params_list
    