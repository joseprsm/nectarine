def get_dataset(config, step: str, target: str):
    def check_name(x):
        return x["name"] == target

    return next(filter(check_name, config[step]["datasets"]))
