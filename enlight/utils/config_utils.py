import sys
import omegaconf
from omegaconf import OmegaConf
from typing import Union


_CLASS_REGISTRY = {}


def add_class_registry(name, class_type):
    assert callable(class_type)
    _CLASS_REGISTRY[name] = class_type


def get_class(path):
    """
    First try to find the class in the registry,
    if it doesn't exist, use importlib to locate it
    """
    if path in _CLASS_REGISTRY:
        return _CLASS_REGISTRY[path]
    try:
        from importlib import import_module

        module_path, _, class_name = path.rpartition(".")
        mod = import_module(module_path)
        try:
            class_type = getattr(mod, class_name)
        except AttributeError:
            raise ImportError(
                "Class {} is not in module {}".format(class_name, module_path)
            )
        return class_type
    except ValueError as e:
        print("Error initializing class " + path, file=sys.stderr)
        raise e


def is_omegaconf(conf):
    return isinstance(conf, omegaconf.Config)


def _is_instantiable(conf):
    return isinstance(conf, (omegaconf.Config, dict)) and 'class' in conf


def recursive_instantiate(conf: Union[OmegaConf, dict], *args, **kwargs):
    if is_omegaconf(conf):
        conf = OmegaConf.to_container(conf, resolve=True)
    assert _is_instantiable(conf)
    if 'params' not in conf:
        conf['params'] = {}
    params = {}
    for key, value in conf['params'].items():
        if _is_instantiable(value):
            value = recursive_instantiate(value)
        params[key] = value
    try:
        class_type = get_class(conf["class"])
        params.update(kwargs)
        return class_type(*args, **params)
    except Exception as e:
        print("Error instantiating {} : {}".format(conf["class"], e), file=sys.stderr)
        raise e


def to_dict(conf):
    if is_omegaconf(conf):
        return OmegaConf.to_container(conf, resolve=True)
    elif isinstance(conf, dict):
        return conf
    else:
        raise NotImplementedError(f'Unknown dict type: {type(conf)}')