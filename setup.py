from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

setup(
    name='kde',
    version='0.0.1',
    description='',
    long_description=readme,
    author='Haluk Acarcicek',
    author_email='acarcicek@outlook.com',
    url='',
    license='',
    packages=["kde"],
    install_requires=["numpy>=1.0"],
    extras_require={
        "dev": [
            "pytest>=6.2.4",
        ]
    },
)