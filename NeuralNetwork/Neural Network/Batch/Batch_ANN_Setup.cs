using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NeuralNetwork
{
    public partial class BatchNeuralNetwork : ArtificialNeuralNetwork
    {
        public BatchInfo batchInfo { get; private set; }

        //ERRORS
        const string ERRORLENGTH = ("<color=red>Output and desired ouput must be the same size</color>");

        //Batch data
        float[][] batchInputs;
        float[][] batchDesiredOutput;
        int[] randomIndices;

        //Sizes
        const int BASEBATCHSIZE = 50;
        int batchSize;
        int sampleInputSize;
        int sampleOuputSize;

        //Cycle through exemples
        int trainingCycleCount = 0;
        int sampleCycleCount = 0;

        int sampleCount = 0;


        public bool isCycleLoop = true;

        public BatchNeuralNetwork(int[] sizeNetwork) : this(sizeNetwork, BASEBATCHSIZE) { }

        public BatchNeuralNetwork(int[] sizeNetwork, int batchSize) : base(sizeNetwork)
        {
            batchInfo = new BatchInfo();

            this.batchSize = batchSize;
            //Error sample can only be the size of the output layer
            sampleOuputSize = sizeNetwork[sizeNetwork.Length - 1];
            sampleInputSize = sizeNetwork[0];

            batchInputs = new float[batchSize][];
            batchDesiredOutput = new float[batchSize][];

            for (int i = 0; i < batchSize; i++)
            {
                batchInputs[i] = new float[sampleInputSize];
                batchDesiredOutput[i] = new float[sampleOuputSize];
            }
        }

        public class BatchInfo
        {
            public int sampleCount;
        }
    }
}
