from setuptools import find_packages, setup


def load_requirements(filename="requirements.txt"):
    with open(filename, "r") as f:
        return f.read().splitlines()


setup(
    name="effiara",
    version="0.1.0",
    description="Package for distributing annotations and calculating annotator agreement/reliability using the EffiARA framework.",  # noqa
    author="Owen Cook",
    author_email="owenscook1@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=load_requirements(),
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
