# Deep-Lighting-Network-ONNX

![demo](Example.JPG)

## Requirements
- Windows 10
- Visual Studio 2019
- CUDA 11.1, Cudnn 8
- Python 3.7
- Torch 1.9.0
- OpenCV 4.5.1 with CUDA

## Installation
- Download Deep-lighting Network and Install required libraries
``` 
git clone https://github.com/WangLiwen1994/DLN 
```

- git clone https://github.com/spacewalk01/Deep-Lighting-Network-ONNX

Open LightingNetwork.sln with Visual Studio 2019
https://docs.microsoft.com/en-us/nuget/quickstart/install-and-use-a-package-in-visual-studio

move save_model.py $DLN_dir/

cd $DLN_dir
python save_model.py --modelfile models/DLN_finetune_LOL.pth


onnxruntime-gpu
