import yaml


class Config(dict):
    def __new__(cls, config_path: str):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
