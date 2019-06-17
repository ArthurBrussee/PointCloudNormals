Normal estimation in point clouds based on [Boulch, 2016] <http://imagine.enpc.fr/~marletr/publi/SGP-2016-Boulch-Marlet.pdf> 'Deep Learning for Robust Normal Estimation in Unstructured Point Clouds' 


## Technique overview

For each point in the point cloud, a potential normal is estimated by picking 3 random points in it's neighbourhoods. All these normals are then summed up in a histogram. In a classical technique, we can simply pick the histogram with the highest count as the 'best' normal. Boulch et al instead use a neural network to regress a histogram to the 'best' hypothesized normal.

The performance is further improved by weighing the chance to pick 3 points by their local densities, by using multiple neighbourhood scales for estimation, and by PCA transforming the normal hypotheses.

## Code Overview 

All C# Unity code for generating normals using max bin and doing CNN inference is in ./Assets/PointCloudCNN. The code for training the network is in ./TrainingCode/. All training & validation data were removed, as well as training checkpoints - you can re-generate them by enabling the Training/Validation data objects in TestScene.unity. 

A pre-trained model is included in ./TrainingCode/saved_models that is used for inference.

The code relies heavily on Unity's new Burst and DOTS functionality.

## Results

![Comparison on Cylinders](./Images/cylinderCompare.png)

![Comparison on Cubes](./Images/cubeCompare.png)

![Comparison on Spheres](./Images/sphereCompare.png)

The above images show reconstruction error with green being a 0 degree error, and red as a 60 degree error. In each image the upper row is using the 'classical' technique, whereas the bottom row is using the neural network. Each column represents a different noise level. As can be seen, the results for the neural network are much better, especially at higher noise levels. 


![Comparison on real world rock mesh](./Images/rockCompare.png)

Comparison on a real world mesh. Left: Using the max bin method. Right: Using the neural network. 

## Running the code

Open the project in Unity 2019.1. Importing might take some time. The CNN is only supported on Windows, and this project was only tested on Windows 10. After opening this project in the Unity Editor, one MUST first _enable_ Jobs->Burst->Enable Compilation, and _disable_ Jobs->Burst->Safety Checks or the max bnin inference will be about a factor ~80 times slower if not just crash.

There are a few different groups that can be toggled on/off to generate different parts of the training data. By default the rock test is enabled, which uses both the max bin method and CNN method, and visualizes the error on some stone meshes. When selecting one of the point clouds you'll see various options, including an option to write training data to a folder.

The neural network inference will currently only work on Windows 10


