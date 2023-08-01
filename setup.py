import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "unet",
    version = "0.1",
    author = "Nghia Huynh",
    author_email = "huynhnguyenhieunghia1999@gmail.com",
    description = ("An package of Image Pretraining using U-Net architecture"),
    packages=['model'],
    long_description=read('README.md'),
)