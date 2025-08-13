_mode = None

def set_mode(mode):
    global _mode

    if mode not in ('local', 'colab'):
        raise ValueError(f"Unsupported mode {mode}. Must either be 'local' or 'colab'.")

    _mode = mode

def get_mode():
    if _mode is None:
        raise RuntimeError("Config mode not set. Call set_mode('local') or set_mode('colab') first.")

    return _mode

def get_data_root():
    if _mode == 'colab':
        return '/content/data/MyDrive/deep-learning-with-pytorch'

    return ''