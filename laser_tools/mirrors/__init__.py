import os
from typing import Dict, Type

from .mirror import Mirror

pkg_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(pkg_dir), "data/mirrors")


def _classname_from_stem(stem: str) -> str:
    return "".join(part.capitalize() for part in stem.split("_") if part)

def get_files(data_dir):
    class_names = []
    filepaths = []

    for filename in os.listdir(data_dir):
        stem = os.path.splitext(filename)[0]
        class_names.append(_classname_from_stem(stem))
        filepaths.append(os.path.join(data_dir, filename))

    return class_names, filepaths

def available():
    class_names, fileaths = get_files(data_dir)
    return class_names

def _register_children() -> Dict[str, Type[Mirror]]:
    
    created: Dict[str, Type[Mirror]] = {}

    if not os.path.isdir(data_dir):
        raise FileNotFoundError("Directory not found.")

    class_names, filepaths = get_files(data_dir)

    for class_name, filepath in zip(class_names, filepaths):

        def __init__(self, _p = filepath):
            Mirror.__init__(self, _p)

        Child = type(class_name, (Mirror, ), {"__init__": __init__, "DATA_PATH": filepath})

        globals()[class_name] = Child
        created[class_name] = Child

    return created

_created = _register_children()

print(*_created.keys())

__all__ = ["Mirror", *_created.keys()]
