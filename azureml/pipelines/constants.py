from pathlib import Path

COMPONENTS_DIR = (Path(__file__).parent.parent / "components").absolute()

ENVIRONMENTS_DIR = (Path(__file__).parent.parent / "environments").absolute()

ENVIRONMENT_FILE = ENVIRONMENTS_DIR / "guidance-env.yaml"

assert COMPONENTS_DIR.exists(), f"Did not find {COMPONENTS_DIR}"
assert ENVIRONMENT_FILE.exists(), f"Did not find {ENVIRONMENT_FILE}"
