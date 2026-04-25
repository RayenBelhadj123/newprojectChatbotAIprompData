from us_housing.paths import DEFAULT_DATASET, PROJECT_ROOT, RAW_DATA_DIR


def test_project_root_points_to_repository() -> None:
    assert (PROJECT_ROOT / "streamlit_app.py").exists()


def test_default_dataset_uses_raw_data_layer() -> None:
    assert RAW_DATA_DIR.name == "01_raw"
    assert DEFAULT_DATASET.parent == RAW_DATA_DIR
    assert DEFAULT_DATASET.name == "us_home_price_analysis_2004_2024.csv"

