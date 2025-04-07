from dotenv import load_dotenv
import os

REQUIRED_ENV_VARS = ["COMET_API_KEY", "COMET_MODE"]


def setup_env():
    load_dotenv()
    for var in REQUIRED_ENV_VARS:
        if os.getenv(var) is None:
            raise ValueError(f"Environment variable {var} is not set")
