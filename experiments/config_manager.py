from dataclasses import dataclass
from typing import List, Optional
import argparse
from pathlib import Path
from config.randomization_config import ExperimentType

@dataclass
class ExperimentConfig:
    controller_types: List[str]
    experiment_type: str
    num_trials: int
    seed: int
    save_trials: bool
    use_serial: bool
    visualize: bool
    when2fail: bool
    max_intensity: float
    intensity_step: float
    trajectory_type: str
    delay_margin: bool = False
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'ExperimentConfig':
        controller_types = args.controller if 'all' not in args.controller else [
            'geo', 'geo-a', 'l1geo', 'indi-a', 'l1mpc', 'mpc', 'xadap'
        ]
        return cls(
            controller_types=controller_types,
            experiment_type=args.experiment,
            num_trials=args.num_trials,
            seed=args.seed,
            save_trials=args.save_trials,
            use_serial=args.serial,
            visualize=args.vis,
            when2fail=args.when2fail,
            max_intensity=args.max_intensity,
            intensity_step=args.intensity_step,
            trajectory_type=args.trajectory,
            delay_margin=args.delay_margin
        )

def parse_experiment_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller', type=str, nargs='+', default=['geo'], 
                       help='controller types: geo, geo-a, l1geo, l1mpc, indi-a, xadap, mpc, all')
    parser.add_argument('--experiment', type=str, default='no', 
                       choices=[e.value for e in ExperimentType],
                       help='experiment type: no, wind, uncertainty, force, torque, rotoreff')
    parser.add_argument('--num_trials', type=int, default=100, help='number of trials to run')
    parser.add_argument('--seed', type=int, default=42, 
                       help='seed for random number generator')
    parser.add_argument('--save_trials', action='store_true',
                       help='save individual trials to csv')
    parser.add_argument('--serial', action='store_true',
                       help='run in serial')
    parser.add_argument('--vis', action='store_true',
                       help='visualize single trial without saving data')
    parser.add_argument('--when2fail', action='store_true',
                       help='find failure point by increasing disturbance intensity')
    parser.add_argument('--max_intensity', type=float, default=10.0,
                       help='maximum intensity multiplier for when2fail mode')
    parser.add_argument('--intensity_step', type=float, default=1,
                       help='intensity increment step for when2fail mode')
    parser.add_argument('--trajectory', type=str, default='random',
                       choices=['random', 'hover', 'circle'],
                       help='trajectory type to use')
    parser.add_argument('--delay_margin', action='store_true',
                       help='calculate delay margin for the given trajectory')
    return ExperimentConfig.from_args(parser.parse_args())