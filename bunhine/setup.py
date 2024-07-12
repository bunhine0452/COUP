from setuptools import setup, find_packages

setup(
    name="bunhine",
    version="0.1",
    packages=find_packages(),
    description="편리한 파이썬 개인 모듈, 데이터 분석 라이브러리",
    author="Bunhine",
    author_email="hb000122@gmail.com",
    url="https://your-url.com",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    ],
)