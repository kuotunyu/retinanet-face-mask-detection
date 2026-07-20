import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "mask_retinanet.yaml")


def load_config(config_path=None):
    """Load the project YAML config and return a dictionary."""
    path = resolve_path(config_path or DEFAULT_CONFIG_PATH)
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required to read config files. Install it with: pip install PyYAML==5.4.1")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config


def cfg_get(config, dotted_key, default=None):
    """Read nested config values using keys such as 'model.input_shape'."""
    current = config
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def resolve_path(path):
    if path is None or path == "":
        return path
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        raise ValueError("Expected a boolean value, got: {}".format(value))
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes", "y", "on"):
        return True
    if normalized in ("false", "0", "no", "n", "off"):
        return False
    raise ValueError("Expected a boolean value, got: {}".format(value))


def parse_int_list(value):
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(v.strip()) for v in value.split(",") if v.strip()]
