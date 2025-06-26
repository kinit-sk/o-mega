import torch
import numpy as np
import os
import sys


class HideOutput:
    stderr = None
    stdout = None
    
    @staticmethod
    def hide() -> None:
        HideOutput.stderr = sys.stderr
        HideOutput.stdout = sys.stdout
        null = open(os.devnull, 'w')
        sys.stdout = sys.stderr = null

    @staticmethod
    def show() -> None:
        sys.stderr = HideOutput.stderr
        sys.stdout = HideOutput.stdout


def get_device() -> torch.device:
    return (
        torch.device("cuda") 
        if torch.cuda.is_available() 
        else torch.device("cpu")
    )


def no_randomness(seed: int = 0) -> None:
    np.random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)