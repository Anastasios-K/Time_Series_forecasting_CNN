import os


def set_environments():
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == "__main__":
    set_environments()
