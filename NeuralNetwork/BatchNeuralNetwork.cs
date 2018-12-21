using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NeuralNetwork
{
    public class BatchNeuralNetwork : NeuralNetwork
    {

        const string ERRORLENGTH = ("<color=red>Output and desired ouput must be the same size</color>");


        float[][] batchOutput;
        float[][] batchDesiredOutput;


        int sampleCycleCount = 0;
        public int sampleCount = 0;
        int sampleSize;
        int trainingCycleCount = 0;

        public BatchNeuralNetwork(int[] sizeNetwork) : this(sizeNetwork, 25)
        {
        }

        public BatchNeuralNetwork(int[] sizeNetwork, int batchSize) : base(sizeNetwork)
        {
            //Error sample can only be the size of the output layer
            sampleSize = sizeNetwork[sizeNetwork.Length - 1];
            batchOutput = new float[batchSize][];
            batchDesiredOutput = new float[batchSize][];

            for (int i = 0; i < batchSize; i++)
            {
                batchOutput[i] = new float[sampleSize];
                batchDesiredOutput[i] = new float[sampleSize];
            }
        }

        public void AddTrainingData(float[] output, float[] desiredOutput)
        {
            if (output.Length != desiredOutput.Length)
            {
                Debug.Log(ERRORLENGTH);
                return;
            }
            for (int i = 0; i < output.Length; i++)
            {
                batchOutput[sampleCycleCount][i] = output[i];
                batchDesiredOutput[sampleCycleCount][i] = desiredOutput[i];
            }
            sampleCycleCount = (sampleCycleCount + 1) % batchDesiredOutput.GetLength(0);
            sampleCount = Mathf.Min(sampleCount + 1, batchDesiredOutput.GetLength(0));
        }
        public string BatchTraining(float learningRate, float momentum)
        {
            if (sampleCount == 0)
                return "";

            SetInput(batchOutput[trainingCycleCount]);
            Update();
            string log = TrainWithExemples(batchDesiredOutput[trainingCycleCount], learningRate, momentum);
            trainingCycleCount = (trainingCycleCount + 1) % sampleCount;
            return log;
        }
        public string BatchTraining(float learningRate)
        {
            return BatchTraining(learningRate, 0);
        }
    }
}