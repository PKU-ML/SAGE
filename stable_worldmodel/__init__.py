"""Public APIs, loaded without initializing unrelated simulator backends."""
from importlib import import_module

__all__ = [
    'World',
    'PlanConfig',
    'pretraining',
    'data',
    'envs',
    'policy',
    'solver',
    'spaces',
    'utils',
    'wm',
    'wrapper',
]


def __getattr__(name):
    attributes = {'World': ('world', 'World'), 'PlanConfig': ('policy', 'PlanConfig'),
                  'pretraining': ('utils', 'pretraining')}
    if name in attributes:
        module, attribute = attributes[name]
        value = getattr(import_module(f'{__name__}.{module}'), attribute)
    elif name in __all__:
        value = import_module(f'{__name__}.{name}')
    else:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    globals()[name] = value
    return value
