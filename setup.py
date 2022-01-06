import sys
from setuptools import setup


def getRequirements():
    with open("requirements.txt", "r") as f:
        read = f.read()

    return read.split("\n")


setup(
    name = 'object and picture overlay',
    version= "1.0.1",
    description='AR with open cv for python',
    long_description='detection of a photo in a video and overlaying it with other pictures, videos or 3d objects',
    author='Mortar Defender',
    license='MIT License',
    url = '__',
    setup_requires = getRequirements(),
    install_requires = getRequirements(),
    include_package_data=True
)
