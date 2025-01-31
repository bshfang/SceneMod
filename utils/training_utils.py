import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import json

import string
import os
import random
import subprocess
import torch


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config
