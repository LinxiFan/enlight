import os
from setuptools import setup, find_packages


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read().strip()


pkg_name = 'enlight'

setup(
    name=pkg_name,
    version='0.0.1',
    author='Jim Fan',
    url='http://github.com/LinxiFan/enlight',
    description='Fast and lightweight vision-based RL framework',
    # long_description=read('README.rst'),
    keywords=['Deep Learning',
              'Reinforcement Learning',
              'Computer Vision'],
    license='mit',
    packages=[
        package for package in find_packages() if package.startswith(pkg_name)
    ],
    entry_points={
        'console_scripts': [
            # 'myexe=enlight.pkg:main',
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Programming Language :: Python :: 3"
    ],
    python_requires='>=3.7',
    include_package_data=True,
    zip_safe=False
)
