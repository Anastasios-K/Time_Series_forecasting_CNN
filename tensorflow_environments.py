import os


def set_environments():
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # any bias addition that is based on tf.nn.bias_add() operates deterministically on GPU
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # forces the selection of deterministic cuDNN convolution and max-pooling algorithms
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide tensorflow automatic comments

    # for further info - nvidia sources:
    # https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_19.06.html


if __name__ == "__main__":
    set_environments()
