from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="ml_project",
    packages=find_packages(),
    version="0.1.0",
    description="HW01 for MADE VK ML in Production Course",
    author="AAKurilovich",
    entry_points={
        "console_scripts": [
            "cli = ml_project.main:main"
        ]
    },
    install_requires=required,
    license="MIT",
)