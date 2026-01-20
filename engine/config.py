from dataclasses import dataclass


@dataclass(frozen=True)
class Tolerances:
    feature_abs_tol: float = 1e-3
    feature_rel_tol: float = 0.02
    tol_emission: float = 1e-3  # mV
    sensitivity_min_changed_features: int = 3


@dataclass(frozen=True)
class CouplingThresholds:
    gap_high: float = 0.01
    spin_high: float = 0.4
    hfcc_high: float = 15.0


@dataclass(frozen=True)
class RunConfig:
    tolerances: Tolerances = Tolerances()
    coupling: CouplingThresholds = CouplingThresholds()
    emission_fixed: bool = False


CONFIG = RunConfig()
