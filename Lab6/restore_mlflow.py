"""Restore the published MLflow run snapshot to a local portable tracking directory."""
import argparse
from pathlib import Path
from zipfile import ZipFile

import yaml


LAB = Path(__file__).resolve().parent


def restore(destination):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    with ZipFile(LAB / "results" / "mlflow_snapshot.zip") as snapshot:
        snapshot.extractall(destination)
    tracking = destination / "mlruns"
    artifacts = destination / "artifacts"
    for experiment in tracking.iterdir():
        meta = experiment / "meta.yaml"
        if not meta.is_file():
            continue
        data = yaml.safe_load(meta.read_text(encoding="utf-8"))
        data["artifact_location"] = artifacts.as_uri()
        meta.write_text(yaml.safe_dump(data), encoding="utf-8")
        for meta in experiment.glob("*/meta.yaml"):
            data = yaml.safe_load(meta.read_text(encoding="utf-8"))
            data["artifact_uri"] = (artifacts / meta.parent.name).as_uri()
            meta.write_text(yaml.safe_dump(data), encoding="utf-8")
    return tracking.as_uri()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=LAB / "runs" / "published")
    args = parser.parse_args()
    print(f'mlflow ui --backend-store-uri "{restore(args.destination)}"')
