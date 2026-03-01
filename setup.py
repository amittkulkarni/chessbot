from setuptools import setup, find_packages

setup(
    name="chessbot",
    version="0.1.0",
    description="Hardware-aware Deep Learning Chess Engine (Post-Training Quantization, ONNX, Supervised MCTS)",
    author="Amit Dilip Kulkarni",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "python-chess",
        "onnxruntime-openvino",
        "torch",
        "numpy",
        "requests",
        "python-dotenv"
    ],
    python_requires=">=3.8",
)