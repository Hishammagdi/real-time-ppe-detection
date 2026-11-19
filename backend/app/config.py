import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    MODEL_WEIGHTS_PATH: str = os.getenv(
        "MODEL_WEIGHTS_PATH", "../model/weights/ppe_best.pt"
    )
    DEFAULT_CONF_THRESHOLD: float = float(os.getenv("DEFAULT_CONF_THRESHOLD", "0.4"))
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "200"))
    HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("BACKEND_PORT", "8000"))
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "*").split(",")


settings = Settings()
