import setuptools
from setuptools import setup

setup(name='standard_precip',
      version='0.4',
      description='Functions to calculate SPI and SPEI',
      url='http://github.com/e-baumer/standard_precip',
      author='Eric Nussbaumer',
      author_email='ebaumer@gmail.com',
      # packages=setuptools.find_packages(),
      license='Apache License 2.0',
      packages=['standard_precip', 'standard_precip/lmoments'],
      install_requires=[
        'numpy>=1.19.0',
        'matplotlib>=3.1.2',
        'scipy>=1.5.4',
        'pandas>=1.1.5',
      ],
      zip_safe=False)
