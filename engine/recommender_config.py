from dataclasses import dataclass


@dataclass(frozen=True)
class SearchConfig:
    radius_primary: float = 8.0
    radius_secondary: float = 12.0
    max_combo_size: int = 3
    beam_width: int = 6
    top_n: int = 5
    safety_strictness: float = 1.0
    allow_secondary_shell: bool = True
    include_second_shell_if_hbond: bool = True


@dataclass(frozen=True)
class ScoreWeights:
    clash_lambda: float = 0.5
    stability_mu: float = 0.7
    confidence_nu: float = 0.3
    synergy_xi: float = 0.2


CONFIG = SearchConfig()
WEIGHTS = ScoreWeights()
