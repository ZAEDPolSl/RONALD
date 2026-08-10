import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run the MRI vessel reporting pipeline.")
    parser.add_argument("--image", type=Path, required=True, help="MRI image file or DICOM directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to data/mri_vessel_reports_example.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    image_path = args.image.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else repo_root / "data" / "mri_vessel_reports_example"
    )
    config_path = output_dir / "config.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "output_dir": str(output_dir),
        "caliber_thresholds_mm": {
            "small": 2.0,
            "large": 5.0,
        },
        "studies": [
            {
                "name": "example_mri",
                "image": str(image_path),
            }
        ],
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "calculate_vesselness_stats.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        cwd=repo_root,
        env=env,
    )


if __name__ == "__main__":
    main()
