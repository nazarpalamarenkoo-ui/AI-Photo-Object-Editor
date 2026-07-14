import torch

from app.config.settings import settings


class DeviceManager:

    @staticmethod
    def get(device: str) -> str:
        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"

        return device