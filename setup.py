from setuptools import setup, Extension

setup(

    name='snn',
    version='0.0.1',
    description='Simulating spiking neural networks',

    author='Clayton Seitz',
    author_email='cwseitz@uchicago.edu',
    ext_modules=[Extension("snn.models", ["snn/models.c"])],
    packages=['snn'])
