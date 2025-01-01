import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stemlab",
    version="0.2.0",
    author="STEM Research",
    author_email="library@stemxresearch.com",
    description="A Python Library for Mathematical and Statistical Computing in STEM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stemxresearch/stemlab",
    license='MIT',
    python_requires='>=3.6',
    include_package_data=True, # # This ensures MANIFEST.in is respected
    install_requires=[
        'scipy',  # Includes numpy, matplotlib, pandas, and other scientific libraries
        'scikit-learn',  # Includes scipy, numpy, and other dependencies
        'plotly',
        'sympy',
        'IPython',
        'statsmodels',
        'kaleido' # for amortization function
    ],
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)