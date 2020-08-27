import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pysan", # Replace with your own username
    version="0.1",
    author="Oliver J. Scholten",
    author_email="oliver@gamba.dev",
    description="Social sequence analysis using Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pysan-dev/pysan",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pandas >= 1.0.3",
        "matplotlib >= 3.2.1",
    ],
    python_requires='>=3.6',
)