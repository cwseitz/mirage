from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

class build_ext(_build_ext):
    """Delay import numpy until build."""
    def finalize_options(self):
        _build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

setup(

    name='snn',
    version='0.0.1',
    description='Simulating spiking neural networks',
    cmdclass={'build_ext': build_ext},
    author='Clayton Seitz',
    author_email='cwseitz@uchicago.edu',
    ext_modules=[Extension("snn.models", ["snn/models.c"])],
    packages=['snn'])
