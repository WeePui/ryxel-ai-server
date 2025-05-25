import importlib
import subprocess
import sys

# Danh sách các package cần thiết
packages = [
    "flask", "opencv-python", "numpy", "onnxruntime", "pandas", "requests",
    "scikit-learn", "underthesea", "mlxtend"
]
def check_and_install_packages(package_list):
    """Kiểm tra và cài đặt các package nếu chưa có."""
    for package in package_list:
        try:
            importlib.import_module(package)
        except ImportError:
            print(f"Package {package} chưa được cài đặt. Đang tiến hành cài đặt...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Gọi hàm để kiểm tra và cài đặt các package
check_and_install_packages(packages)