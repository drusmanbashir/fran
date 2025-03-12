import setuptools
import os
with open(os.path.join(os.path.dirname(__file__), "requirements.txt"), encoding="utf-8") as f:
    all_requirements = f.read().splitlines()
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

development_only = ["gudhi", "nurbspy"]

# Filter out development-only requirements using partial string matching
requirements = [
    req for req in all_requirements 
    if not any(dev_req in req for dev_req in development_only)
]


setuptools.setup(
    name="fran",  # Package name
    version="0.8.0",  # Initial release version
    author="Usman Bashir",  # Author name set to yours
    author_email="usman.bashir@example.com",  # Replace with your actual email if needed
    description="A Python package for interfacing with or providing functionality for fran.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fran",  # Replace with your repository URL
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Update if using a different license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
)

