Deep Point cloud normal estimation using hough transforms


## Overview 

All C# Unity code for generating normals using max bin and doing CNN inference is in ./Assets/PointCloudCNN. The code for training the network is in ./PointCloudCNN/. The training & validation data were removed, as well as training checkpoints. A pre-trained model is included in PointCloudCNN/saved_models that is used for inference.

## Running the code

Open the project in Unity 2019.1. Importing might take some time. The CNN is only supported on Windows, and this project was only tested on Windows 10. After opening this project in the Unity Editor, one MUST first _enable_ Jobs->Burst->Enable Compilation, and _disable_ Jobs->Burst->Safety Checks or the max bnin inference will be about a factor ~80 times slower if not just crash.

There are a few different groups that can be toggled on/off to generate different parts of the training data. By default the rock test is enabled, which uses both the max bin method and CNN method, and visualizes the error on some stone meshes. When selecting one of the point clouds you'll see various options, including an option to write training data to a folder.