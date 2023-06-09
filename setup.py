from setuptools import setup

setup(
    name="wasserstein_filter",
    version="0.1.0",
    description="Variational Filtering via Wasserstein Gradient Flow",
    author="Hany Abdulsamad, Adrien Corenflos",
    author_email="hany@robot-learning.de",
    install_requires=[
        "numpy",
        "scipy",
        "jax",
        "jaxlib",
        "jaxopt",
        "typing_extensions",
        "matplotlib",
        "dask"
    ],
    packages=["wasserstein_filter"],
    zip_safe=False,
)
