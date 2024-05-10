from setuptools import setup, find_packages

VERSION = "0.1a"
DESCRIPTION = "One-way coupled Lagrangian Particle Tracking for CFD simulations. Includes many auxiliary tools."
LONG_DESCRIPTION = "Detailed description of the package can be requested from the author at this moment."

# Setting up
setup(
    # name must match the folder name
    name="project-arrakis",
    version=VERSION,
    url="https://github.com/kalagotla/project-arrakis",
    license="MIT",
    author="kal @ Dilip Kalagotla",
    author_email="<dilipkalagotla@gmail.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],
    keywords=["python", "first package"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
