import importlib.util
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

    package_init = repo_path / "__init__.py"
    inference_path = repo_path / "inference.py"
    if not package_init.exists():
        raise FileNotFoundError(f"LobePrior package init not found at {package_init}")
    if not inference_path.exists():
        raise FileNotFoundError(f"LobePrior inference module not found at {inference_path}")

    package_name = "vendor_lobeprior"
    if package_name not in sys.modules:
        package_spec = importlib.util.spec_from_file_location(
            package_name,
            package_init,
            submodule_search_locations=[str(repo_path)],
        )
        if package_spec is None or package_spec.loader is None:
            raise ImportError(f"Could not load LobePrior package from {package_init}")
        package_module = importlib.util.module_from_spec(package_spec)
        sys.modules[package_name] = package_module
        package_spec.loader.exec_module(package_module)

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
