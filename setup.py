import setuptools

setuptools.setup(name='swin',
      version='0.1',
      description='The funniest joke in the world',
      url='https://github.com/sanyalsc/MLIA-2022-final',
      author='MLIA Fall 2022 SWIN UNENT group',
      author_email='sanyalster@gmail.com',
      license='MIT',
      packages=setuptools.find_packages("src"),
      package_dir={'':"src"},
      install_requires=[
        'torch',
        'pillow',
        'numpy',
        'monai',
        'einops',
        'scikit-image',
        'tqdm'
      ])