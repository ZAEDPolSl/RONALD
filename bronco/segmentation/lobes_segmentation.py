import importlib.util
import types
import sys
from pathlib import Path


def lobes_segmentation(sitk_image, config=None):
    config = {} if config is None else dict(config)
    repo_path = Path(
        config.get("lobeprior_repo")
        or Path(__file__).resolve().parents[2] / "vendor" / "LobePrior"
    ).expanduser().resolve()

    if not repo_path.exists():
        raise FileNotFoundError(f"LobePrior submodule not found at {repo_path}")

    inference_path = repo_path / "inference.py"
    if not inference_path.exists():
        raise FileNotFoundError(f"LobePrior inference module not found at {inference_path}")

    package_name = "vendor_lobeprior"
    if package_name not in sys.modules:
        package_module = types.ModuleType(package_name)
        package_module.__path__ = [str(repo_path)]
        sys.modules[package_name] = package_module

    # Compatibility shims for older LobePrior modules that still use absolute
    # imports like `utils.general` or `model.unet_diedre`.
    alias_paths = {
        "utils": repo_path / "utils",
        "model": repo_path / "model",
    }
    for alias, alias_path in alias_paths.items():
        if alias not in sys.modules and alias_path.exists():
            alias_module = types.ModuleType(alias)
            alias_module.__path__ = [str(alias_path)]
            sys.modules[alias] = alias_module

    module_name = f"{package_name}.inference"
    module = sys.modules.get(module_name)
    if module is None:
        inference_spec = importlib.util.spec_from_file_location(module_name, inference_path)
        if inference_spec is None or inference_spec.loader is None:
            raise ImportError(f"Could not load LobePrior inference module from {inference_path}")
        module = importlib.util.module_from_spec(inference_spec)
        sys.modules[module_name] = module
        inference_spec.loader.exec_module(module)

    use_prior = config.get("use_prior", True)
    return module.predict_lobes_in_memory(sitk_image, use_prior=use_prior)
