import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

with open('requirements.txt') as fin:
    lines = fin.readlines()
    lines = [o.strip() for o in lines]
    lines = [o for o in lines if len(o) > 0]
    req = [o for o in lines if not o.startswith('#') and not o.startswith('git+')]

setup(
    name = "resvit",
    version = "0.1",
    author = "Nghia Huynh",
    author_email = "huynhnguyenhieunghia1999@gmail.com",
    description = ("An package of Image Pretraining using U-Net architecture"),
    packages=['resvit'],
    long_description=read('README.md'),
)