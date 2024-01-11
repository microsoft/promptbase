import logging
import pathlib


from azure.ai.ml import load_component, MLClient, load_environment
from azure.ai.ml.entities import Component, Environment

from constants import COMPONENTS_DIR, ENVIRONMENT_FILE


_logger = logging.getLogger(__file__)
_logger.setLevel(logging.INFO)

# This dictionary lists the attributes to be added to ComponentCollector
ALL_COMPONENTS = dict(
    jsonl_embeddings="jsonl_embeddings_aoai_component.yaml",
    jsonl_guidance="jsonl_guidance_component.yaml",
    jsonl_key_filter="jsonl_key_filter_component.yaml",
    jsonl_key_rename="jsonl_key_rename_component.yaml",
    jsonl_knn_cosine_similarity="jsonl_knn_cosine_similarity_component.yaml",
    jsonl_mmlu_fetch="jsonl_mmlu_fetch_component.yaml",
    jsonl_random_examples="jsonl_random_examples_component.yaml",
    jsonl_schema_checker="jsonl_schema_checker_component.yaml",
    jsonl_score_multiplechoice="jsonl_score_multiplechoice_component.yaml",
    jsonl_to_json="jsonl_to_json_component.yaml",
    uri_folder_to_file="uri_folder_to_file_component.yaml",
)


def create_component_from_yaml(
    ml_client: MLClient,
    yaml_path: pathlib.Path,
    version_string: str,
    environment: Environment = None,
) -> Component:
    _logger.info(f"Loading {yaml_path}")
    loaded_yaml = load_component(source=yaml_path)
    _logger.info("Changing version")
    loaded_yaml.version = version_string
    _logger.info("Changing environment")
    loaded_yaml.environment = environment
    _logger.info("Creating component")
    my_comp = ml_client.components.create_or_update(loaded_yaml)
    _logger.info(f"Component {my_comp.name}:{my_comp.version} created")
    return my_comp


def create_environment_from_yaml(
    ml_client: MLClient, yaml_path: pathlib.Path, version_string: str
) -> Environment:
    _logger.info(f"Loading {yaml_path}")
    loaded_yaml = load_environment(source=yaml_path)
    _logger.info("Changing version")
    loaded_yaml.version = version_string
    _logger.info("Creating Environment")
    my_env = ml_client.environments.create_or_update(loaded_yaml)
    _logger.info(f"Environment {my_env.name}:{my_env.version} created")
    return my_env


class ComponentCollector:
    def __init__(
        self,
        ml_client: MLClient,
        component_base_dir: pathlib.Path,
        version_string: str,
    ):
        self._client = ml_client
        self._base_dir = component_base_dir
        self._version_string = version_string

    def prepare(self):
        _logger.info(f"Creating environment")
        component_environment = create_environment_from_yaml(
            self._client, ENVIRONMENT_FILE, self._version_string
        )
        for attr_name, component_string in ALL_COMPONENTS.items():
            assert not hasattr(self, attr_name)
            _logger.info(f"Creating {component_string} from YAML")
            component = create_component_from_yaml(
                self._client,
                self._base_dir / component_string,
                environment=component_environment,
                version_string=self._version_string,
            )
            _logger.info(f"Adding attribute {attr_name}")
            setattr(self, attr_name, component)
        _logger.info("Added all components")


def get_component_collector(
    ml_client: MLClient, version_string: str
) -> ComponentCollector:
    components = ComponentCollector(ml_client, COMPONENTS_DIR, version_string)
    components.prepare()

    return components
