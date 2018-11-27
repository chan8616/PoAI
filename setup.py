from setuptools import setup
import os

setup(name='TensorflowGUI',
    version='0.1',
    packages=['model', 'icons', 'utils', 'gui'],
    scripts=['bin/TensorflowGUI'],
    install_requires=[
        'tensorflow==1.8',
        'wxpython', 
        'Pillow',
        'matplotlib',
        'sklearn'],
    python_requires='>3')
