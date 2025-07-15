import os
from pathlib import Path

from setuptools import find_packages, setup

root_dir = Path(__file__).parent
long_description = (root_dir / "README.md").read_text(encoding="utf-8")


def load_requirements(filename="requirements.txt"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    req_file = os.path.join(base_dir, filename)
    with open(req_file, "r") as f:
        return f.read().splitlines()


setup(
    name="effiara",
    version="0.1.1",
    description="Package for distributing annotations and calculating annotator agreement/reliability using the EffiARA framework.",  # noqa
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Owen Cook",
    author_email="owenscook1@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=load_requirements(),
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
