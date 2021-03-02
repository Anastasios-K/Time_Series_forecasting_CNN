import os


def clean_non_test_files() -> None:
    """
    Remove all non-test files from the current directory.

    """
    valid_files = list(filter(lambda file:
                              file.startswith("test") or file.startswith("clean"),
                              os.listdir()))

    [os.remove(file) for file in os.listdir() if file not in valid_files]


clean_non_test_files()
