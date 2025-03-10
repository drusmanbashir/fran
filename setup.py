import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

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
    python_requires=">=3.11",
    install_requires=[
        "torch",  # No specific version to ensure compatibility with different CUDA versions
        "torchvision",  # No specific version to allow flexibility
        "torchaudio",
        "batchgenerators==0.25",
        "connected_components_3d==3.17.0",
        # "cudf==0.6.1.post1",
        # "cugraph==0.6.1.post1",
        # "cuml==0.6.1.post1",
        "dicom_utils>=0.1.0",
        "fastapi==0.115.11",
        "fastcore==1.6.3",
        # "grad_cam==1.5.3",
        # "gudhi==3.10.1",
        "h5py==3.12.1",
        "ipdb==0.13.13",
        "itk==5.4.2.post1",
        "lightning==2.5.0.post0",
        "lightning_utilities==0.11.9",
        "matplotlib==3.10.1",
        "monai==1.4.0",
        "neptune==1.13.0",
        "numpy",
        "nurbspy==1.1.2",
        "opencv_python==4.10.0.84",
        "opencv_python_headless==4.10.0.84",
        "openpyxl==3.1.5",
        "pandas==2.2.3",
        "paramiko==3.5.0",
        "Pillow==11.1.0",
        "plotly==6.0.0",
        "prefetch_generator==1.0.3",
        "psutil==6.0.0",
        "pydicom==2.3.0",
        "pytest==8.3.3",
        "PyYAML==6.0.2",
        "ray==2.36.1",
        "reportlab==4.2.5",
        "requests==2.32.3",
        "roboflow==1.1.49",
        "scipy==1.15.2",
        "seaborn==0.13.2",
        "Send2Trash==1.8.3",
        "setuptools==75.1.0",
        "SimpleITK==2.4.1",
        "scikit-image",
        "supervision==0.25.1",
        "tqdm==4.66.5",
        "ultralytics==8.3.44",
        "xnat>=0.7.0",
    ],
)

