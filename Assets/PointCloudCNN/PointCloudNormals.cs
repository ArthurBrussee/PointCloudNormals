using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TC;
using TensorFlow;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.Profiling;
using Random = Unity.Mathematics.Random;
using KNN;
using KNN.Jobs;

/// <summary>
/// Main class to handle generation of the point cloud normals / training data
/// </summary>
public static class PointCloudNormals {
	// Hyper parameters, see report for more details
	const int c_m = 33;
	const int c_kDens = 5;

	// const float Alpha = 0.95f;
	// const float Epsilon = 0.073f;
	// Could be calculated from Alpha & Epsilon above but better to just fix it here
	const int c_hypotheses = 1000;

	// Base number of k-neighbourhood to consider
	const int c_baseK = 100;
	const int c_scaleLevels = 5;

	// K levels to use at different scales
	static readonly int[] KScales = {c_baseK / 4, c_baseK / 2, c_baseK, c_baseK * 2, c_baseK * 4};

	static ProfilerMarker s_generateDataMarker;
	static ProfilerMarker s_queryNeighboursMarker;
	static ProfilerMarker s_houghTexMarker;

	/// <summary>
	/// Hough histogram data. Uses a byte to hold accumulation value. This is fine for a low number of Hypotheses (H much smaller than M * M *255)
	/// </summary>
	public unsafe struct HoughHistogram {
#pragma warning disable 649
		public fixed byte Counts[c_m * c_m * c_scaleLevels];
#pragma warning restore 649

		public float3x3 NormalsBasis;
		public float3x3 TexBasis;

		public void GetScaledHisto(float[] space) {
			// Scale each layer seperately
			for (int k = 0; k < c_scaleLevels; ++k) {
				// Count up total tex weights
				float maxVal = 0.0f;

				for (int index = 0; index < c_m * c_m; ++index) {
					int offsetIndex = k * c_m * c_m + index;
					maxVal = math.max(maxVal, Counts[offsetIndex]);
				}

				for (int index = 0; index < c_m * c_m; ++index) {
					int offsetIndex = k * c_m * c_m + index;
					space[offsetIndex] = Counts[offsetIndex] / maxVal;
				}
			}
		}
	}

	/// <summary>
	/// Main job for generating the hough histograms
	/// </summary>
	[BurstCompile(CompileSynchronously = true)]
	unsafe struct HoughHistogramsJob : IJobParallelFor {
		[ReadOnly] public NativeArray<float3> Positions;
		[ReadOnly] public NativeArray<float> PointDensities;
		[ReadOnly] public NativeArray<int> Neighbours;
		[ReadOnly] public NativeArray<float3> TrueNormals;

		[NativeDisableParallelForRestriction]
		public NativeArray<HoughHistogram> HoughHistograms;
		public Random Rand;

		int PickWeightedRandomVal(float* a, int kVal) {
			float maxVal = a[kVal - 1];
			float value = Rand.NextFloat(maxVal);

			for (int i = 0; i < kVal; i++) {
				if (value <= a[i]) {
					return i;
				}
			}

			// Should never get here...
			return -1;
		}

		/// <summary>
		/// Try get normal from 3 points. Fails if points are co-linear
		/// </summary>
		bool TryGetNormal(int p0, int p1, int p2, out float3 normal) {
			normal = math.cross(Positions[p2] - Positions[p0], Positions[p1] - Positions[p0]);

			// If we pick bad points or co-tangent points, we end up with a bad 
			// 0 length normal... can't use this, discard this hypothesis and try again
			if (math.lengthsq(normal) < math.FLT_MIN_NORMAL) {
				return false;
			}

			// Success!
			normal = math.normalize(normal);
			return true;
		}

		// Run job, calculate multi-scale texture for a point
		public void Execute(int index) {
			int kMax = KScales[c_scaleLevels - 1];
			int neighbourBaseOffset = index * kMax;

			// First, determine PCA basis
			// We do this for one scale - our medium scale
			const int c_kBasisIndex = 2;
			
			// Pick neighbours up to kVal
			int kVal = KScales[c_kBasisIndex];

			// Construct covariance matrix
			float3x3 posCovariance = float3x3.identity;

			float3 meanPos = float3.zero;
			for (int i = 0; i < kVal; ++i) {
				meanPos += Positions[Neighbours[neighbourBaseOffset + i]];
			}
			meanPos /= kVal;

			for (int dimi = 0; dimi < 3; ++dimi) {
				for (int dimj = 0; dimj < 3; ++dimj) {
					float cov = 0.0f;

					for (int i = 0; i < kVal; ++i) {
						float3 testPos = Positions[Neighbours[neighbourBaseOffset + i]];
						cov += (testPos[dimi] - meanPos[dimi]) * (testPos[dimj] - meanPos[dimj]);
					}

					cov /= kVal;
					posCovariance[dimi][dimj] = cov;
				}
			}

			// Eigen decompose the covariance matrix. This gives 3 eigen values in D
			// And eigen basis. Faster than a full blown PCA solver
			mathext.eigendecompose(posCovariance, out float3x3 pcaBasis, out float3 _);

			// Store all 2D hypothesized normals in hough space
			NativeArray<float2> houghNormals = new NativeArray<float2>(c_hypotheses, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
			
			float totalDensity = 0.0f;

			// Count up probabilities
			float* cumProb = stackalloc float[kMax];
			for (int i = 0; i < kMax; ++i) {
				int neighbourIndex = Neighbours[neighbourBaseOffset + i];
				totalDensity += PointDensities[neighbourIndex];
				cumProb[i] = totalDensity;
			}
			
			int hypothesesCompleted = 0;
			while (hypothesesCompleted < c_hypotheses) {
				// Pick 3 points, weighted by the local density
				int n0 = PickWeightedRandomVal(cumProb, kVal);
				int n1 = PickWeightedRandomVal(cumProb, kVal);
				int n2 = PickWeightedRandomVal(cumProb, kVal);
				
				int p0 = Neighbours[neighbourBaseOffset + n0];
				int p1 = Neighbours[neighbourBaseOffset + n1];
				int p2 = Neighbours[neighbourBaseOffset + n2];

				if (!TryGetNormal(p0, p1, p2, out float3 hypNormal)) {
					continue;
				}

				// Normally the sign ambiguity in the normal is resolved by facing towards the 
				// eg. Lidar scanner. In this case, since these point clouds do not come from scans
				// We cheat a little and resolve the ambiguity by instead just comparing to ground-truth
				if (math.dot(TrueNormals[index], hypNormal) < 0) {
					hypNormal = -hypNormal;
				}

				houghNormals[hypothesesCompleted] = math.mul(pcaBasis, hypNormal).xy;
				hypothesesCompleted++;
			}

			// Now do 2D PCA for the hough spaced normals
			float3x3 texCov = float3x3.identity;
			float2 meanTex = float2.zero;
			
			for (int i = 0; i < c_hypotheses; ++i) {
				meanTex += houghNormals[i];
			}

			// Average 2D hough position
			meanTex /= c_hypotheses;

			for (int dimi = 0; dimi < 2; ++dimi) {
				for (int dimj = 0; dimj < 2; ++dimj) {
					float cov = 0.0f;
					for (int i = 0; i < c_hypotheses; ++i) {
						float2 testPos = houghNormals[i];
						cov += (testPos[dimi] - meanTex[dimi]) * (testPos[dimj] - meanTex[dimj]);
					}

					cov /= c_hypotheses;
					texCov[dimi][dimj] = cov;
				}
			}
			texCov[2][2] = math.FLT_MIN_NORMAL; // ensure z has smallest eigen value, so we only rotate xy
			
			mathext.eigendecompose(texCov, out float3x3 pcaBasisTex, out float3 _);

			// Get a pointer to the current hough histogram
			ref HoughHistogram histogram = ref UnsafeUtilityEx.ArrayElementAsRef<HoughHistogram>(HoughHistograms.GetUnsafePtr(), index);
			
			// Setup the basis
			histogram.NormalsBasis = pcaBasis;
			histogram.TexBasis = pcaBasisTex;
			
			// Generate a texture at each scale
			for (int kIndex = 0; kIndex < c_scaleLevels; ++kIndex) {
				// Pick neighbours up to kVal
				kVal = KScales[kIndex];
				hypothesesCompleted = 0;

				while (hypothesesCompleted < c_hypotheses) {
					// Pick 3 points, weighted by the local density
					int n0 = PickWeightedRandomVal(cumProb, kVal);
					int n1 = PickWeightedRandomVal(cumProb, kVal);
					int n2 = PickWeightedRandomVal(cumProb, kVal);
					
					int p0 = Neighbours[neighbourBaseOffset + n0];
					int p1 = Neighbours[neighbourBaseOffset + n1];
					int p2 = Neighbours[neighbourBaseOffset + n2];

					if (!TryGetNormal(p0, p1, p2, out float3 hypNormal)) {
						continue;
					}

					hypothesesCompleted++;

					// Normally the sign ambiguity in the normal is resolved by facing towards the 
					// eg. Lidar scanner. In this case, since these point clouds do not come from scans
					// We cheat a little and resolve the ambiguity by instead just comparing to ground-truth
					if (math.dot(TrueNormals[index], hypNormal) < 0) {
						hypNormal = -hypNormal;
					}

					float2 localHough = math.mul(pcaBasisTex, math.mul(pcaBasis, hypNormal)).xy;

					float2 uv = (localHough + 1.0f) / 2.0f;
					int2 histogramPos = math.int2(math.floor(uv * c_m));

					// When either x or y is exactly 1 the floor returns M which is out of bounds 
					// So clamp tex coordinates
					histogramPos = math.clamp(histogramPos, 0, c_m - 1);

					// Now vote in the histogram
					int histoIndex = histogramPos.x + histogramPos.y * c_m + kIndex * c_m * c_m;
					
					// Clamp so we don't overflow (in rare cases)
					if (histogram.Counts[histoIndex] < byte.MaxValue) {
						histogram.Counts[histoIndex]++;
					}
				}
			}
		}
	}

	/// <summary>
	/// Generate training data and visualize as TC Cloud
	/// </summary>
	public static NativeArray<HoughHistogram> CalculateHistograms(NativeArray<MeshPoint> meshPoints, NativeArray<float3> queryPositions, NativeArray<float3> trueNormals) {
		// Performance measure markers
		s_generateDataMarker = new ProfilerMarker("GenerateDataMarker");
		s_queryNeighboursMarker = new ProfilerMarker("QueryNeighbours");
		s_houghTexMarker = new ProfilerMarker("HoughTexture");
		
		using (s_generateDataMarker.Auto()) {
			int maxK = KScales[c_scaleLevels - 1];
			s_queryNeighboursMarker.Begin();

			var truePositions = new NativeArray<float3>(meshPoints.Select(p => p.Position).ToArray(), Allocator.TempJob);
			var neighbours = new NativeArray<int>(truePositions.Length * maxK, Allocator.TempJob);

			var tree = new KnnContainer(truePositions, true, Allocator.TempJob);

			// Get all neighbours for each point
			new KNearestBatchQueryJob(tree, truePositions, neighbours).ScheduleBatch(truePositions.Length, 512).Complete();

			var pointDensities = new NativeArray<float>(meshPoints.Length, Allocator.TempJob);
			
			new QueryDensityJob {
				PointDensitiesResult = pointDensities,
				Positions = truePositions,
				Neighbours = neighbours,
				MaxK = maxK
			}.Schedule(truePositions.Length, 256).Complete();
			
			// All done, dispose memory used for temp results
			tree.Dispose();
			
			s_queryNeighboursMarker.End();
			
			// Now construct all hough histograms
			var houghTextures = new NativeArray<HoughHistogram>(queryPositions.Length, Allocator.Persistent);
			
			using (s_houghTexMarker.Auto()) {
				var random = new Random(23454575);

				// Schedule job to create hough histograms
				var createHoughTexJob = new HoughHistogramsJob {
					Positions = truePositions,
					PointDensities = pointDensities,
					TrueNormals = trueNormals,
					Neighbours = neighbours,
					Rand = random,
					HoughHistograms = houghTextures,
				};
				
				createHoughTexJob.Schedule(houghTextures.Length, 256).Complete();
			}

			// Release all memory we used in the process
			neighbours.Dispose();
			pointDensities.Dispose();
			meshPoints.Dispose();
			truePositions.Dispose();
			return houghTextures;
		}
	}

	public static NativeArray<float> CalculateScore(NativeArray<float3> reconstructedNormals, NativeArray<float3> trueNormalsSample, out float rms, out float pgp) {
		var reconstructionError = new NativeArray<float>(reconstructedNormals.Length, Allocator.TempJob);
		rms = 0.0f;
		pgp = 0;
		for (int i = 0; i < reconstructedNormals.Length; ++i) {
			float angle = math.degrees(math.acos(math.dot(reconstructedNormals[i], trueNormalsSample[i])));
			reconstructionError[i] = angle;
			rms += angle * angle;

			if (angle < 8) {
				pgp += 1.0f;
			}
		}

		pgp /= reconstructedNormals.Length;
		rms = math.sqrt(rms / reconstructedNormals.Length);
		return reconstructionError;
	}

	public static NativeArray<float3> CalculateNormals(NativeArray<HoughHistogram> houghTextures, NativeArray<float3> trueNormalsSample, bool useNeuralNet) {
		NativeArray<float3> reconstructedNormals;
		if (!useNeuralNet) {
			// Use classical methods
			reconstructedNormals = EstimateNormals(houghTextures, trueNormalsSample);
		} else {
			// Pass to CNN!
			EstimatePropertiesCnn(houghTextures, trueNormalsSample, out reconstructedNormals);
		}

		return reconstructedNormals;
	}

	public static PointCloudData ConstructPointCloudData(NativeArray<float3> positions, NativeArray<float3> normals, NativeArray<Color32> albedo, bool showError, NativeArray<float> reconstructionError) {
		// TC Particles point cloud data for visualization
		PointCloudData pointCloudData = ScriptableObject.CreateInstance<PointCloudData>();
		
		Color32[] cloudAlbedoValues;
		// Show either albedo color, or reconstruction error
		if (showError) {
			var errorGradient = new Gradient();

			// Populate the color keys at the relative time 0 and 1 (0 and 100%)
			var colorKey = new GradientColorKey[2];
			colorKey[0].color = Color.green;
			colorKey[0].time = 0.0f;
			colorKey[1].color = Color.red;
			colorKey[1].time = 1.0f;
			errorGradient.colorKeys = colorKey;

			cloudAlbedoValues = reconstructionError.Select(p => {
				float val = math.saturate(math.abs(p) / 60.0f);
				Color col = errorGradient.Evaluate(val);
				return (Color32) col;
			}).ToArray();
		} else {
			cloudAlbedoValues = albedo.ToArray();
		}

		Vector3[] cloudPoints = positions.Select(p => new Vector3(p.x, p.y, p.z)).ToArray();
		Vector3[] cloudNormals = normals.Select(v => new Vector3(v.x, v.y, v.z)).ToArray();

		// Write data to TC Particles 
		pointCloudData.Initialize(
			cloudPoints,
			cloudNormals,
			cloudAlbedoValues,
			1.0f,
			Vector3.zero,
			Vector3.zero
		);
		return pointCloudData;
	}

	public static void WriteTrainingImages(string folder, string prefix, NativeArray<HoughHistogram> histograms, NativeArray<float3> trueNormals) {
		string imagesPath = "PointCloudCNN/" + folder + "/";
		Directory.CreateDirectory(imagesPath);

		var fileName = new StringBuilder();

		// Lay out K channels next to each other in the channel
		var saveTex = new Texture2D(c_m, c_m * c_scaleLevels);
		var colors = new Color32[c_m * c_m * c_scaleLevels];

		float[] scratchSpace = new float[c_m * c_m * c_scaleLevels];

		// Write training data to disk
		for (int i = 0; i < histograms.Length; ++i) {
			HoughHistogram testHistogram = histograms[i];

			// Get scaled histogram
			testHistogram.GetScaledHisto(scratchSpace);

			// Write to color array
			for (int index = 0; index < c_scaleLevels; ++index) {
				byte level = (byte) math.clamp((int) math.round(255 * scratchSpace[index]), 0, 255);
				colors[index] = new Color32(level, level, level, 255);
			}

			saveTex.SetPixels32(colors);

			// Get ground truth normal in hough texture space 
			float2 setNorm = math.mul(testHistogram.TexBasis, math.mul(testHistogram.NormalsBasis, trueNormals[i])).xy;

			// Write PNG to disk
			fileName.Clear();
			fileName.Append(imagesPath);
			fileName.Append(prefix);
			fileName.Append(i);
			fileName.Append("_x_");
			fileName.Append(setNorm.x.ToString("0.000"));
			fileName.Append("_y_");
			fileName.Append(setNorm.y.ToString("0.000"));
			fileName.Append(".png");
			File.WriteAllBytes(fileName.ToString(), saveTex.EncodeToPNG());
		}
	}

	[BurstCompile(CompileSynchronously = true)]
	struct QueryDensityJob : IJobParallelFor {
		public NativeArray<float> PointDensitiesResult;

		[ReadOnly] public NativeArray<float3> Positions;
		[ReadOnly] public NativeArray<int> Neighbours;

		public int MaxK;
		
		public void Execute(int index) {
			int kDensIndex = Neighbours[index * MaxK + c_kDens];
			PointDensitiesResult[index] = math.distancesq(Positions[index], Positions[kDensIndex]);
		}
	}

	static float3 GetNormalFromPrediction(float2 prediction, float3x3 normalsBasis, float3x3 texBasis, float3 trueNormal) {
		float normalZ = math.sqrt(math.saturate(1.0f - math.lengthsq(prediction)));
		float3 normal = math.normalize(math.float3(prediction, normalZ));
		float3 trueNormalBasis = math.mul(texBasis, math.mul(normalsBasis, trueNormal));
		
		// Normally the sign ambiguity in the normal is resolved by facing towards the 
		// eg. Lidar scanner. In this case, since these point clouds do not come from scans
		// We cheat a little and resolve the ambiguity by instead just comparing to ground-truth
		if (math.dot(normal, trueNormalBasis) < 0) {
			normal.z *= -1;
		}

		float3x3 invTex = math.transpose(texBasis);
		float3x3 invNormals = math.transpose(normalsBasis);
		return math.mul(invNormals, math.mul(invTex, normal));
	}

	/// <summary>
	/// Job to estimate an array of normals using the max bin method
	/// </summary>
	[BurstCompile(CompileSynchronously = true)]
	struct EstimateNormalsJob : IJobParallelFor {
		[ReadOnly] public NativeArray<HoughHistogram> Textures;
		[ReadOnly] public NativeArray<float3> TrueNormals;

		public NativeArray<float3> Normals;

		public void Execute(int index) {
			float3 trueNormal = TrueNormals[index];
			var tex = Textures[index];

			unsafe {
				var totalHisto = new NativeArray<int>(c_m * c_m, Allocator.Temp);

				for (int kIndex = 0; kIndex < c_scaleLevels; ++kIndex) {
					for (int y = 0; y < c_m; ++y) {
						for (int x = 0; x < c_m; ++x) {
							totalHisto[x + y * c_m] += tex.Counts[x + y * c_m + kIndex * c_m * c_m];
						}
					}
				}

				int2 maxBin = math.int2(-1, -1);
				int maxBinCount = 0;
				for (int y = 0; y < c_m; ++y) {
					for (int x = 0; x < c_m; ++x) {
						int count = totalHisto[x + y * c_m];

						if (count > maxBinCount) {
							maxBinCount = count;
							maxBin = math.int2(x, y);
						}
					}
				}

				// M - 1 to compensate for floor() in encoding
				// This way with M=33 we encode cardinal directions exactly!
				float2 normalUv = math.float2(maxBin) / (c_m - 1.0f);
				float2 normalXy = 2.0f * normalUv - 1.0f;

				// Write final predicted normal
				Normals[index] = GetNormalFromPrediction(normalXy, tex.NormalsBasis, tex.TexBasis, trueNormal);
			}
		}
	}

	/// <summary>
	/// Estimate normals for array of hough tex.
	/// </summary>
	static NativeArray<float3> EstimateNormals(NativeArray<HoughHistogram> textures, NativeArray<float3> trueNormals) {
		// First calculate normals for every bin
		var normals = new NativeArray<float3>(textures.Length, Allocator.Persistent);
		var job = new EstimateNormalsJob {
			Textures = textures,
			Normals = normals,
			TrueNormals = trueNormals
		};

		job.Schedule(textures.Length, 256).Complete();
		return normals;
	}
	
	/// <summary>
	/// Estimate normals & roughness using the trained CNN
	/// </summary>
	static void EstimatePropertiesCnn(NativeArray<HoughHistogram> histograms, NativeArray<float3> trueNormals, out NativeArray<float3> normals) {
		// Construct tensofrflow graph
		var graphData = File.ReadAllBytes("PointCloudCNN/saved_models/tf_model.pb");

		using (var graph = new TFGraph()) {
			graph.Import(graphData);

			// Return arrays
			var normalsRet = new NativeArray<float3>(histograms.Length, Allocator.Persistent);

			const int c_chunkCount = 500;

			// Calculate 1000 points at a time
			// That means our imageTensor is about ~20MB!
			// Doing all points at once just crashes because the allocation is so large...
			int chunks = (int) math.ceil(histograms.Length / (float) c_chunkCount);
			float[,,,] imageTensor = new float[c_chunkCount, c_m, c_m, KScales.Length];

			Profiler.BeginSample("Construct session");
			// Create network query
			var session = new TFSession(graph);
			Profiler.EndSample();

			for(int chunk = 0; chunk < chunks; ++chunk) {
				Profiler.BeginSample("Construct tensor");
				int start = chunk * c_chunkCount;
				int end = math.min(histograms.Length, (chunk + 1) * c_chunkCount);
				int count = end - start;

				// Write our histograms to the image tensor
				Parallel.For(0, count, tensorIndex => {
					float[] histoSpace = new float[c_m * c_m * c_scaleLevels];

					int i = tensorIndex + start;
					histograms[i].GetScaledHisto(histoSpace);

					for (int k = 0; k < c_scaleLevels; ++k) {
						for (int y = 0; y < c_m; ++y) {
							for (int x = 0; x < c_m; ++x) {
								imageTensor[tensorIndex, c_m - y - 1, x, k] = histoSpace[x + y * c_m + k * c_m * c_m];
							}
						}
					}
				});

				Profiler.EndSample();

				// Now run query in tensorflow
				Profiler.BeginSample("Run TF Query");
				var runner = session.GetRunner();
				var inputNode = graph["input_input"][0];
				var outputNode = graph["output/BiasAdd"][0]; // Keras for some reason splits output node in a few sub-nodes. Grab final output value after bias add
				runner.AddInput(inputNode, imageTensor);
				runner.Fetch(outputNode);
				var output = runner.Run();
				Profiler.EndSample();

				// Write to results array
				Profiler.BeginSample("Write results");
				// Fetch the results from output as 2D tensor
				float[,] predictions = (float[,]) output[0].GetValue();

				for (int tensorIndex = 0; tensorIndex < count; ++tensorIndex) {
					int i = start + tensorIndex;
					var tex = histograms[i];

					float2 predNormal = math.float2(predictions[tensorIndex, 0], predictions[tensorIndex, 1]);
					normalsRet[i] = GetNormalFromPrediction(predNormal, tex.NormalsBasis, tex.TexBasis, trueNormals[i]);
				}
				Profiler.EndSample();
			}

			normals = normalsRet;
		}
	}
}