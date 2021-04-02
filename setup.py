import pathlib
from setuptools import setup, find_packages


README = (pathlib.Path(__file__).parent / "README.md").read_text()


setup(
    name="easy-tensorflow",
    version="0.0.1",
    author="Fernando Nieuwveldt",
    author_email="fdnieuwveldt@gmail.com",
    description="An interface containing easy tensorflow model building blocks and feature pipelines",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/fernandonieuwveldt/easyflow",
    packages=["easyflow"],
)
