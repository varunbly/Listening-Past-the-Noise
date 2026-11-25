import os
from pathlib import Path


def course_data(root):
    return Path(os.environ.get("PROJECT4_DATA", root / "project4"))
