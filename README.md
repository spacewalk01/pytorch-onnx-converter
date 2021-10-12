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
- Clone Deep-lighting Network and Install required libraries
``` 
git clone https://github.com/WangLiwen1994/DLN 
```

You can skip step 1 in case model.onnx is already in Project folder.
### Step 1
- Clone this project
```
git clone https://github.com/spacewalk01/Deep-Lighting-Network-ONNX
```
- Move onnx generator to DLN folder and run it
```
move save_model.py $DLN_folder 
cd $DLN_folder
python save_model.py --modelfile models/DLN_finetune_LOL.pth
```
### Step 2
- Open LightingNetwork.sln with Visual Studio 2019
- 
https://docs.microsoft.com/en-us/nuget/quickstart/install-and-use-a-package-in-visual-studio

move save_model.py $DLN_dir/

cd $DLN_dir
python save_model.py --modelfile models/DLN_finetune_LOL.pth
- Select x64 and Release 
- Right-click on Project and Open to Manage NuGet Packages
- Search onnxruntime-gpu and install it by clicking down-arrow.
- Setup Opencv on Properties
- Build and run the project
