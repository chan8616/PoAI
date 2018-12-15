from setuptools import setup
import os
import pip
from pip._internal import main
main(['install', '-U', '-f',
    'https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-16.04',
    'wxPython'])

setup(name='TensorflowGUI',
    version='0.1',
    packages=['model', 'icons', 'utils', 'gui'],
    scripts=['bin/TensorflowGUI'],
    install_requires=[
        'h5py',
        'Pillow',
        'matplotlib',
        'sklearn'],
    python_requires='>=3.6')
