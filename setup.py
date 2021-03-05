from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()


setup(
    name="easyflow",
    version="0.0.0",
    author="Fernando Nieuwveldt",
    author_email="fdnieuwveldt@gmail.com",
    description="A package containing easy tensorflow model building",
    long_description=readme,
    url="https://github.com/fernandonieuwveldt/easyflow",
    packages=find_packages(),
)
