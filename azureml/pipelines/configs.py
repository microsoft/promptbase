from typing import Dict, List, Union

from dataclasses import dataclass, field


@dataclass
class AMLConfig:
    workspace_name: str = str()
    resource_group: str = str()
    subscription_id: str = str()


@dataclass
class AOAIConfig:
    endpoint: str = str()
    model: str = str()
    compute_target: str = str()


@dataclass
class ZeroShotRunConfig:
    base_experiment_name: str = str()
    tags: Dict[str, str] = field(default_factory=dict)
    mmlu_dataset: str = str()
    mmlu_split: str = str()
    default_compute_target: str = str()
    guidance_program: str = str()
    aoai_config: AOAIConfig = field(default_factory=AOAIConfig)