import time
import torch
import pycuda.driver as cuda


def wait_for_gpu_memory(
    device: torch.device,
    required_memory,
    max_wait_time = 10,
    gap_time = 10,
    max_continuous_count = 3,
):
    """
    Args:
        device: torch.device 需要使用的显卡设备
        required_memory: 运行所需显存,单位 GB
        max_wait_time: 最大等待时间,单位小时
        gap_time: 查询间隔时间
        max_continuous_count: 连续查询到满足所需显存次数以退出
    """
    device_id = int(str(device).split(':')[1])
    required_memory = required_memory * 1024 * 1024 * 1024
    max_wait_time = max_wait_time * 3600
    continuous_count = 0
    wait_start = time.time()
    
    cuda.init()
    device = cuda.Device(device_id)
    context = device.make_context()

    while True:
        # 查询显存使用情况
        free_memory = cuda.mem_get_info()[0]
        
        print(f'free_memory: {free_memory / 1024 / 1024 / 1024:.2f} GB')
        
        if free_memory >= required_memory:
            continuous_count += 1
            if continuous_count >= max_continuous_count:
                context.pop()
                print(f'gpu memory is enough.')
                return
        else:
            continuous_count = 0
            
        # 检查是否超时
        if time.time() - wait_start > max_wait_time:
            context.pop()
            print("wait for gpu memory timeout, exit")
            exit()
            
        time.sleep(gap_time)


def wait_for_multi_gpu(
    devices: list[int],
    required_memory,
    max_wait_time=10,
    gap_time=10,
    max_continuous_count=3
):
    """
    Args:
        devices: 整数列表，包含需要检查的GPU设备ID
        required_memory: 运行所需显存，单位 GB
        max_wait_time: 最大等待时间，单位小时
        gap_time: 查询间隔时间
        max_continuous_count: 连续查询到满足所需显存次数以退出
    """
    required_memory = required_memory * 1024 * 1024 * 1024
    max_wait_time = max_wait_time * 3600
    continuous_count = 0
    wait_start = time.time()

    cuda.init()

    while True:
        all_ready = True
        for device_id in devices:
            device_obj = cuda.Device(device_id)
            context = device_obj.make_context()

            free_memory = cuda.mem_get_info()[0]

            print(f'GPU {device_id}: free_memory: {free_memory / 1024 / 1024 / 1024:.2f} GB')

            if free_memory < required_memory:
                all_ready = False

            context.pop()
            
        if all_ready:
            continuous_count += 1
        else:
            continuous_count = 0

        if continuous_count >= max_continuous_count:
            print('All specified GPUs have enough memory.')
            return

        if time.time() - wait_start > max_wait_time:
            print("wait for gpu memory timeout, exit")
            exit()

        time.sleep(gap_time)
