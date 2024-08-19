'''
命令行参数解析
试着运行： python arg_parse.py --size=10
'''

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 添加参数
    parser.add_argument('--size', type=int, default=0)

    args = parser.parse_args()

    print(f'args.size: {args.size}')
