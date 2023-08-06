from setuptools import setup, find_packages

VERSION = "0.1"
DESCRIPTION = "A Lagrangian Particle Tracking package"
LONG_DESCRIPTION = "Includes a set of tools for Lagrangian Particle Tracking like search, interpolation, etc."

# Setting up
setup(
    # name must match the folder name
    name="project-arrakis",
    version=VERSION,
    author="kal @ Dilip Kalagotla",
    author_email="<dilipkalagotla@gmail.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'
    keywords=["python", "first package"],
    classifiers=[
        "Development Status :: Work in Progress",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
