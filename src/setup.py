from setuptools import setup, find_packages

setup(
    name="promptbase",
    version="0.1.0",
    author="Microsoft",
    description="Advanced prompting for advanced intelligence",
    # url="https://github.com/repo",  # Replace with the URL of your project
    packages=find_packages(),
    install_requires=[
        "datasets",
        "tqdm",
        "openai",
        "python-liquid",
        "GitPython",
        "torch",
        "scikit-learn",
    ],
    python_requires=">=3.9",  # Specify the minimum Python version required
)
