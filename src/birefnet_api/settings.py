from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    birefnet_local_path: Path = Path("data/pretrained/birefnet")
    birefnet_model_id: str = "ZhengPeng7/BiRefNet"
    device: str = "cuda"
