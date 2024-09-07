import os

import yaml

# Load configuration from config.yaml
config_file_path = "config.yaml"
try:
    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)
        print("Configuration file loaded successfully")
except (yaml.YAMLError, FileNotFoundError) as exc:
    print(f"Error loading configuration file: {exc}")
    raise

# Retrieve settings from config
db_path = config["database"]["path"]
table_name = config["database"]["table_name"]
model_name = config["model"]["name"]
pretrained_model = config["model"]["pretrained_model"]
