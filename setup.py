"""Package configuration for the tvrdart module."""

from setuptools import setup, find_packages

setup(name='tvrdart',
      version='1.1',
      description='TVR-DART implementation',
      author='Folkert Bleichrodt',
      author_email='3368283-3chelon@users.noreply.gitlab.com',
      packages=find_packages(),
      install_requires=[
          'numpy', 'scipy', 'matplotlib',
          'sklearn', 'imageio',
      ],
      zip_safe=False)
