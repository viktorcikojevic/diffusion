from pathlib import Path
from diffusion.envs.environments import DATA_DUMPS_DIR, DATA_DIR, MODEL_OUT_DIR

REPO_DIR = Path(__file__).absolute().resolve().parent.parent.parent.parent


DATA_DIR = Path(DATA_DIR)
DATA_DUMPS_DIR = Path(DATA_DUMPS_DIR)
MODEL_OUT_DIR = Path(MODEL_OUT_DIR)


if __name__ == "__main__":
    pass
