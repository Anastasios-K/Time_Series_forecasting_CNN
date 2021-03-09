import os
import shutil


def clean_non_test_files() -> None:
    """ Remove all non-test files and folders from the current directory. """
    valid_items = list(filter(lambda file:
                              file.startswith("test") or file.startswith("clean"),
                              os.listdir()))

    [shutil.rmtree(item) for item in os.listdir() if item not in valid_items and os.path.isdir(item)]
    [os.remove(file) for file in os.listdir() if file not in valid_items]


a = clean_non_test_files()
