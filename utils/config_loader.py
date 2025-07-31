import os
import yaml

def load_config():
    # print exactly what file it's opening
    project_root = os.path.dirname(os.path.dirname(__file__))
    cfg_path = os.path.join(project_root, 'config.yaml')
    #print(f" Looking for config at: {cfg_path}")

    # read and show what's inside
    with open(cfg_path, 'r', encoding='utf-8-sig') as f:
        config_text = f.read()
    #print("Raw config file contents:\n", config_text)

    # load it
    config = yaml.safe_load(config_text)
    #print("YAML keys loaded:", config.keys())

    return config

