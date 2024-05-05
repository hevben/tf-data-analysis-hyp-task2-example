import pandas as pd
import numpy as np
from hyppo.ksample import MMD

chat_id = 310598863 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    alpha = 0.08
    s,p = MMD(compute_kernel = "laplacian").test(x,y)
    return p<alpha
