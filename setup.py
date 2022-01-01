import sys
from setuptools import setup


def getRequirements():
    with open("requirements.txt", "r") as f:
        read = f.read()

    return read.split("\m")


setup(
    name = 'AR for open cv',
    version= "1.0.1",
    description='__',
    long_description='__',
    author='Matthew Matl',
    license='MIT License',
    url = 'https://github.com/mmatl/pyrender',
    setup_requires = getRequirements,
    install_requires = getRequirements,
    include_package_data=True
)
