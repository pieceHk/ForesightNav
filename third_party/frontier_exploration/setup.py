import setuptools

setuptools.setup(
    name="frontier_exploration",
    version="0.1",
    author="Naoki Yokoyama",
    author_email="naokiyokoyama@github",
    description="Frontier_exploration for habitat",
    url="https://github.com/naokiyokoyama/frontier_exploration",
    packages=["frontier_exploration", *["frontier_exploration." + name for name in setuptools.find_packages()]],
    package_dir={"frontier_exploration": "."},
)
