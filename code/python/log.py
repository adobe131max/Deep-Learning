'''
日志打印
输出到终端同时保存到日志文件中
'''
import logging
from datetime import datetime

def setup_logger(log_file=datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log', level=logging.DEBUG):
    """
    设置日志记录器，使其同时输出到终端并保存到文件中。

    :param log_file: 日志文件的路径，默认是 开始时间 exp: 
    :param level: 日志级别，默认是 logging.DEBUG
    :return: 配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger()
    logger.setLevel(level)

    # 创建处理器 - 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # 创建处理器 - 文件处理器
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(level)

    # 创建日志格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 将格式器添加到处理器
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# 使用示例
logger = setup_logger()
logger.debug('这是一个调试信息')
logger.info('这是一个信息')
logger.warning('这是一个警告')
logger.error('这是一个错误')
logger.critical('这是一个严重错误')

