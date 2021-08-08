import pathlib
import io
import os
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text(encoding='utf-8')
REQUIREMENTS = (pathlib.Path(__file__).parent / "requirements.txt").read_text().splitlines()[1:]
REQUIRES_PYTHON = '>=3.6.0'


setup(
    name="easy-tensorflow",
    version="1.1.1",
    author="Fernando Nieuwveldt",
    author_email="fdnieuwveldt@gmail.com",
    description="An interface containing easy tensorflow model building blocks and feature pipelines",
    long_description_content_type='text/markdown',
    long_description=README,
    url="https://github.com/fernandonieuwveldt/easyflow",
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude='tests'),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
