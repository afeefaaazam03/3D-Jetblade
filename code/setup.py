from setuptools import find_packages, setup

setup(name='bladedelta', version='1.1.0', description='Evidence-bounded 3D damage estimation on turbine blades',
      packages=find_packages(include=['bladedelta']), python_requires='>=3.10',
      install_requires=['torch>=2.1', 'numpy', 'scipy', 'opencv-python-headless', 'pillow'], license='MIT')
