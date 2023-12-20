from typing import Dict, List, Union

from dataclasses import dataclass, field


@dataclass
class AMLConfig:
    workspace_name: str = str()
    resource_group: str = str()
    subscription_id: str = str()
