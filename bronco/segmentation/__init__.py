from importlib import import_module

__all__ = [
    "airways_segmentation",
    "lobes_segmentation",
    "lungs_segmentation",
    "trachea_main_bronchus_segmentation",
    "vessel_segmentation",
    "blobs_segmentation",
    "mediastinum_segmentation",
]

_EXPORTS = {
    "airways_segmentation": (".airways_segmentation", "airways_segmentation"),
    "lobes_segmentation": (".lobes_segmentation", "lobes_segmentation"),
    "lungs_segmentation": (".lungs_segmentation", "lungs_segmentation"),
    "trachea_main_bronchus_segmentation": (
        ".trachea_segmentation",
        "trachea_main_bronchus_segmentation",
    ),
    "vessel_segmentation": (".vessel_segmentation", "vessel_segmentation"),
    "blobs_segmentation": (".blobs_segmentation", "blobs_segmentation"),
    "mediastinum_segmentation": (
        ".mediastinum_segmentation",
        "mediastinum_segmentation",
    ),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
