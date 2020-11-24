from setuptools import find_packages, setup

setup(name="downloader",
      packages=find_packages(),
        entry_points = {'console_scripts': ['downloader=downloader.downloader:main']})
