from pathlib import Path

# Reusable project root path
ROOT_DIR = Path(__file__).resolve().parents[3]

def get_project_path(*parts):
    """
    Convenience function to get a path relative to the project root.
    Usage: get_project_path("data", "raw", "output.csv")
    """
    return ROOT_DIR.joinpath(*parts)

# Common folders
DATA_DIR = get_project_path("data")
RAW_DIR = get_project_path("data", "raw")
INTERIM_DIR = get_project_path("data", "interim")
PROCESSED_DIR = get_project_path("data", "processed")
MODELS_DIR = get_project_path("models")
RESULTS_DIR = get_project_path("results")
FIGURES_DIR = get_project_path("results", "figures")

