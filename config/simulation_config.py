from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from pathlib import Path

@dataclass
class SimulationConfig:
    """Configuration for simulation runs"""
    world: Any
    vehicles: List[Any]
    controllers: List[Any]
    wind_profiles: List[Any]
    trajectories: List[Any]
    num_simulations: int = 100
    ext_force: Optional[List[Any]] = None
    ext_torque: Optional[List[Any]] = None
    parallel: bool = True
    save_individual_trials: bool = False
    save_trial_path: Optional[Path] = None
    experiment_type: str = 'no'
    output_file: Optional[Path] = None
    controller_type: Optional[str] = None
    world_size: float = 10.0