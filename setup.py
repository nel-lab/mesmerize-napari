from setuptools import setup, find_packages

setup(
    name="mesmerize-napari",
    version="0.0",
    packages=find_packages(),
    url="https://github.com/nel-lab/mesmerize-napari",
    license="Apache",
    author="Kushal Kolar, Arjun Putcha",
    author_email="",
    description="Mesmerize-like CaImAn functionality for napari",
    entry_points={"napari.plugin": "Mesmerize = mesmerize_napari"},
)
