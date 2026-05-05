import importlib.util
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

    spec = importlib.util.spec_from_file_location("lobeprior_inference", inference_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load LobePrior inference module from {inference_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    use_prior = config.get("use_prior", True)
    return module.predict_lobes_in_memory(sitk_image, use_prior=use_prior)
