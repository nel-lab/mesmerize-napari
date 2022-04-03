import os
from pathlib import Path
import json
from inspect import getmembers
from typing import *
from .utils import HOME


class _Config:
    recent_parent_paths: List[Path] = None
    recent_batch_paths: List[Path] = None

    def __init__(self):
        config_dir = Path(os.environ[HOME], '.mesmerize-napari')

        if not config_dir.is_dir():
            os.makedirs(config_dir, exist_ok=True)

        self._config_path = config_dir.joinpath('config.json')
        
        self._config_keys = [k[0] for k in getmembers(self) if not k[0].startswith('_')]

        if not self._config_path.is_file():
            self._write()

        with open(self._config_path, 'r') as cf:
            config = json.load(cf)
            for k in config.keys():
                setattr(self, k, config[k])

    def _write(self):
        config = dict()
        for k in self._config_keys:
            config[k] = getattr(self, k)

        with open(self._config_path, 'w') as cf:
            json.dump(config, cf)

    def __setattr__(self, key, value):
        super(_Config, self).__setattr__(key, value)
        if not key.startswith('_'):
            self._write()


CONFIG = _Config()

