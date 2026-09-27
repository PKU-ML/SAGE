"""World-model APIs; importing one backbone need not load dataset services."""
from importlib import import_module

_EXPORTS = {
    'SIGReg': 'loss', 'VCReg': 'loss', 'PLDMLoss': 'loss',
    'TemporalStraighteningLoss': 'loss',
    'load_pretrained': 'utils', 'save_pretrained': 'utils',
    'GCRL': 'gcrl', 'PreJEPA': 'prejepa',
    'LeWM': 'lewm', 'MultiViewLeWM': 'lewm',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(f'{__name__}.{_EXPORTS[name]}'), name)
    globals()[name] = value
    return value
