from setuptools import setup

setup(name='standard_precip',
      version='0.1',
      description='Functions to calculate SPI and SPEI',
      url='http://github.com/e-baumer/standard_precip',
      author='Eric Nussbaumer',
      author_email='ebaumer@gmail.com',
      license='Apache License 2.0',
      packages=['standard_precip','standard_precip.tests'],
      zip_safe=False)