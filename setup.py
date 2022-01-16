from setuptools import setup, find_packages

setup(
    name='caiman-napari-prototype',
    version='0.0',
    packages=find_packages(),
    url='',
    license='',
    author='kushal',
    author_email='',
    description='caiman plugin for napari',
    entry_points={'napari.plugin': 'Caiman-CNMF = caiman_napari'}
)
