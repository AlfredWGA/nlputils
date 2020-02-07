import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nlputils", # Replace with your own username
    version="1.0.0",
    author="AlfredWGA",
    author_email="guoao.wei@outlook.com",
    description="A package containing utilities for Natural Language Processing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlfredWGA/nlputils",
    keywords='NLP, Natural Language Processing',
    packages=setuptools.find_packages(),   # Find packages from current directory.
    # If packages are placed in a distint folder such as `src`, 
    # `package_dir` param need to be included.
    # package_dir={"": "src"},     
    include_package_data=True, # Include everything in the package, else __init__.py only.
    python_requires='>=3.6',
    zip_safe=False
)