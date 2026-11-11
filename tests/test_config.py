import copy

import pytest

from dmrec.config import REPO_ROOT, load_config, resolve_path, validate_config


def test_default_config_loads():
    config = load_config()
    assert config["seed"] == 42
    assert config["lsh"]["bands"] * config["lsh"]["rows"] == config["minhash"]["num_perm"]


def test_resolve_path_puts_csv_under_raw_dir():
    config = load_config()
    assert resolve_path(config, "recipes_csv") == REPO_ROOT / "data/foodcom/RAW_recipes.csv"
    assert resolve_path(config, "processed_dir") == REPO_ROOT / "data/processed"


def test_band_row_mismatch_is_rejected():
    config = copy.deepcopy(load_config())
    config["lsh"]["rows"] += 1
    with pytest.raises(ValueError):
        validate_config(config)
