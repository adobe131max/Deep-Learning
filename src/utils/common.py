def round_floats(obj, n=4):
    """
    将要保存为 json 的数据中的所有小数只保留 n 位小数
    
    Args:
        n: 要保存的有效位数
    Example:
        >>> json.dump(round_floats(info), f, indent=4)
    """
    if isinstance(obj, float):
        return round(obj, n)
    elif isinstance(obj, dict):
        return {key: round_floats(value, n) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(item, n) for item in obj]
    else:
        return obj