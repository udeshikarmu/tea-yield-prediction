"""
Register the project data with DagsHub Data Engine.

DVC pushes file *contents* to DagsHub storage, but the repository's "Datasets"
tab only lists datasets created through Data Engine. This script:
  1. creates (or reuses) a Data Engine datasource pointing at the repo's data/ folder
     (covers the DVC-tracked raw CSV and the processed temporal splits), and
  2. saves it as a named dataset so it appears in the "Datasets" tab.

Idempotent: safe to run on every merge to master.
"""
import os
import sys
import yaml


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def register_dataset():
    token = os.getenv("DAGSHUB_TOKEN")
    if not token:
        print("[DagsHub] DAGSHUB_TOKEN not set - skipping dataset registration.")
        return 0

    from dagshub.auth import add_app_token
    from dagshub.data_engine import datasets, datasources
    from dagshub.data_engine.model.errors import DatasetNotFoundError

    add_app_token(token)

    tracking = load_config()["tracking"]
    repo = tracking["dagshub_repo"]
    datasource_name = tracking["dagshub_datasource"]
    dataset_name = tracking["dagshub_dataset"]

    ds = datasources.get_or_create(repo=repo, name=datasource_name, path="data")
    print(f"[DagsHub] Datasource ready: '{datasource_name}' -> {repo}/data")

    try:
        datasets.get_dataset(repo, dataset_name)
        print(f"[DagsHub] Dataset '{dataset_name}' already exists - nothing to do.")
    except DatasetNotFoundError:
        ds.save_dataset(dataset_name)
        print(f"[DagsHub] Saved dataset '{dataset_name}'.")

    print(f"[DagsHub] View it at https://dagshub.com/{repo}/datasets")
    return 0


if __name__ == "__main__":
    sys.exit(register_dataset())
