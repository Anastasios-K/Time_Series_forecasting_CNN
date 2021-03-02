import os


valid_files = list(filter(lambda file:
                          file.startswith("test") or file.startswith("__init__"),
                          os.listdir()))

[os.remove(file) for file in os.listdir if file not in valid_files]