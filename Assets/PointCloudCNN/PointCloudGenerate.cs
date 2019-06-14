using TC;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using Random = Unity.Mathematics.Random;

/// <summary>
/// Interface for point cloud mesh
/// </summary>
public class PointCloudGenerate : MonoBehaviour {
	public GameObject Tester;

	public int PointCount = 10000;

	public string CloudName = "TestCloud";
	public string Folder = "TrainingData";
	public bool WriteData;

	public bool ShowErrors;
	
	public bool UseCNN;
	
	public float NoiseLevel = 0.01f;
	public float SampleRate = 1.0f;

	// Start is called before the first frame update
	void Start() {
		Mesh mesh = Tester.GetComponent<MeshFilter>().sharedMesh;
		Material mat = Tester.GetComponent<MeshRenderer>().sharedMaterial;

		var tex = mat.GetTexture("_MainTex") as Texture2D;
		var smoothnessTex = mat.GetTexture("_MetallicGlossMap") as Texture2D;
		var normalTex = mat.GetTexture("_BumpMap") as Texture2D;

		if (tex == null) {
			tex = Texture2D.whiteTexture;
		}

		if (smoothnessTex == null) {
			smoothnessTex = Texture2D.whiteTexture;
		}

		// Get points on the mesh
		var meshPoints = MeshSampler.SampleRandomPointsOnMesh(mesh, tex, smoothnessTex, normalTex, PointCount, NoiseLevel);

		// Pick what particles we're going to actually calculate normals for
		var sampleIndices = new NativeList<int>(Allocator.TempJob);
		var rand = new Random(8976543);

		for (int i = 0; i < meshPoints.Length; ++i) {
			if (rand.NextFloat() <= SampleRate) {
				sampleIndices.Add(i);
			}
		}

		// Get properties of sampled particles
		var queryPositions = new NativeArray<float3>(sampleIndices.Length, Allocator.TempJob);
		var trueNormals = new NativeArray<float3>(sampleIndices.Length, Allocator.TempJob);
		var queryColors = new NativeArray<Color32>(sampleIndices.Length, Allocator.TempJob);

		for (int i = 0; i < sampleIndices.Length; ++i) {
			int index = sampleIndices[i];
			queryPositions[i] = meshPoints[index].Position;
			trueNormals[i] = meshPoints[index].Normal;
			queryColors[i] = meshPoints[index].Albedo;
		}
		
		var histograms = PointCloudNormals.CalculateHistograms(meshPoints, queryPositions, trueNormals);

		
		// Folder, CloudName, WriteData, UseCNN, ShowErrors

		// Now that we have the hough histograms,
		// we can estimate normals!

		NativeArray<float3> reconstructedNormals = PointCloudNormals.CalculateNormals(histograms, trueNormals, UseCNN);

		// Ok we have our properties -> Measure how well we did...

		// Measure RMS & PGP per particle
		NativeArray<float> reconstructionError = PointCloudNormals.CalculateScore(reconstructedNormals, trueNormals, out float rms, out float pgp);
		
		
		// string methodName = UseCNN ? "CNN" : "MaxBin";
		// Debug.Log($"{name} finished using {methodName}. Total RMS: {rms}, PGP: {pgp}. Time: {sw.ElapsedMilliseconds}ms");
		var pointCloudData = PointCloudNormals.ConstructPointCloudData(queryPositions, reconstructedNormals, queryColors, ShowErrors, reconstructionError);

		// Write hough textures to disk if requested
		if (WriteData) {
			PointCloudNormals.WriteTrainingImages(Folder, CloudName, histograms, trueNormals);
		}

		queryPositions.Dispose();
		trueNormals.Dispose();
		sampleIndices.Dispose();
		queryColors.Dispose();
		reconstructedNormals.Dispose();
		reconstructionError.Dispose();
		
		var system = GetComponent<TCParticleSystem>();
		system.Emitter.PointCloud = pointCloudData;
		system.Emitter.Emit(pointCloudData.PointCount);
		GetComponent<MeshRenderer>().enabled = false;
	}
}