# from setuptools import find_packages, setup

# setup(
#     name='src',
#     packages=find_packages(),
#     version='0.1.0',
#     description='A short description of the project.',
#     author='Jatin Gupta',
#     license='MIT',
# )





import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
