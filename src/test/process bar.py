import time

for i in range(101):
    time.sleep(0.1)
    print(f"\r进度：[{('=' * i).ljust(100)}] {i/100:.0%}", end='')  # fuck yellow box!
