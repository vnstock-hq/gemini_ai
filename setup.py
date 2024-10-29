import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gemini_ai",
    version="0.0.5",
    author="Thinh Vu",
    author_email="vnstock.hq@gmail.com",
    description="A Python package for the Gemini AI generative model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vnstock-hq/gemini-ai",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "google-generativeai",
        "pillow",
        "ipywidgets",
        "ipython",
        "pytest",
        "pytest-cov",
    ],
    include_package_data=True,
)
