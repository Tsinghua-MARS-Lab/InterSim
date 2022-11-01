from setuptools import setup

setup(name='interactive_sim',
      version='0.3',
      # install_requires=['gym==0.18.3', 'tk', 'Cython', 'tqdm'],
      install_requires=['gym==0.18.3', 'Cython', 'tqdm'],
      packages = ['interactive_sim'])
      # install_requires=['gym', 'shapely', 'pyquaternion'])  # for NuScenes
