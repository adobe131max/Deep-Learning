import time

from tqdm import tqdm

"""
tqdm 显示进度条
"""

for _ in tqdm(range(100)):
    time.sleep(0.001)
