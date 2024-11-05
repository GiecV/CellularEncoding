from setuptools import setup, find_packages

setup(
    name="CellularEncoding",
    version="0.1",
    # Assuming code is in the `src` folder
    packages=find_packages(),
    package_dir={"": "."},  # Source code folder
)
