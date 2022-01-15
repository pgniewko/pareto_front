from setuptools import setup, find_packages
import versioneer

setup(
    name="pareto_front",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Pareto Front on GPU",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "torch>=1.6.0",
        "numpy",
        "xmltodict",
    ],
    include_package_data=True,
)
