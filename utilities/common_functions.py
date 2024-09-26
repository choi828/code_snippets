import yaml

def load_config(config_path='config.yaml'):
    """
    Load the configuration from a YAML file.

    Args:
        config_path (str): Path to the config file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config