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

    repo_path_str = str(repo_path)
    if repo_path_str not in sys.path:
        sys.path.insert(0, repo_path_str)

    from inference import predict_lobes_in_memory

    use_prior = config.get("use_prior", True)
    return predict_lobes_in_memory(sitk_image, use_prior=use_prior)
