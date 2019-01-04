using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NeuralNetwork
{
    public partial class BatchNeuralNetwork : ArtificialNeuralNetwork
    {


        /// <summary>
        /// Add one training data to the batch.
        /// Adding more training data will give more exemple.
        /// </summary>
        /// <param name="inputs">The current inputs.</param>
        /// <param name="desiredOutput">The desired output.</param>
        public void AddTrainingData(float[] inputs, float[] desiredOutput, float weight = 1)
        {
            if (inputs.Length != SizeNetwork[0] || desiredOutput.Length != outputSize)
            {
                Debug.LogError(ERRORLENGTH);
                Debug.LogError("Current input size = " + inputs.Length + ", size network input = " + SizeNetwork[0]);
                Debug.LogError("Desired output size = " + desiredOutput.Length + ", size network output = " + outputSize);

                return;
            }
            if (!isCycleLoop && sampleCount == batchSize)
            {
                //Got all the samples
                //will learn over those samples
                return;
            }

            for (int i = 0; i < inputs.Length; i++)
                batchInputs[sampleCycleCount][i] = inputs[i];

            for (int i = 0; i < desiredOutput.Length; i++)
                batchDesiredOutput[sampleCycleCount][i] = desiredOutput[i];

            batchWeight[sampleCycleCount] = weight;

            sampleCycleCount = (sampleCycleCount + 1) % batchSize;
            sampleCount = Mathf.Min(sampleCount + 1, batchSize);
            batchInfo.sampleCount = sampleCount;
            UpdateRandomIndices();
        }

        void UpdateRandomIndices()
        {
            randomIndices = new int[sampleCount];
            for (int i = 0; i < sampleCount; i++)
            {
                randomIndices[i] = i;
            }
            randomIndices.Shuffle();
        }

        /// <summary>
        /// <para>
        /// Train the neural network using all the exemples
        /// previously added with the method AddTrainingData.
        /// </para>
        /// <para>
        /// For best results, call this method, 
        /// then set the new inputs and update the neural network.
        /// </para>
        /// 
        /// </summary>
        /// <param name="learningRate">Learning rate.</param>
        /// <param name="momentum">Momentum.</param>
        public void BatchTraining(float learningRate = 0.1f, float momentum = 0)
        {
            if (sampleCount == 0)
                return;

            int index = randomIndices[trainingCycleCount];
            float weightedLearningRate = learningRate * batchWeight[index];

            SetInput(batchInputs[index]);
            Update();
            TrainWithExemples(batchDesiredOutput[index], weightedLearningRate, momentum);

            trainingCycleCount++;
            if (trainingCycleCount == sampleCount)
            {
                trainingCycleCount = 0;
                randomIndices.Shuffle();
            }
        }
    }
}