using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NeuralNetwork
{
    public partial class ArtificialNeuralNetwork
    {
        /// <summary>
        /// Trains the neural network with exemples.
        /// </summary>
        /// <returns>The log (optional)</returns>
        /// <param name="targetOutputs">Target outputs.</param>
        /// <param name="learningRate">Learning rate.</param>
        public void TrainWithExemples(float[] targetOutputs, float learningRate)
        {
            TrainWithExemples(targetOutputs, learningRate, 0);
        }

        /// <summary>
        /// Trains the neural network with exemples.
        /// </summary>
        /// <returns>The log (optional)</returns>
        /// <param name="targetOutputs">Target outputs.</param>
        /// <param name="learningRate">Learning rate.</param>
        /// <param name="momentum">Momentum.</param>
        public void TrainWithExemples(float[] targetOutputs, float learningRate, float momentum)
        {
            if (targetOutputs.Length > sizeNetwork[sizeNetwork.Length - 1])
                Debug.Log(ERRORLENGTH);

            CalculateSetError(targetOutputs);
            Train(learningRate, momentum);
        }

        void Train(float learningRate)
        {
            Train(learningRate, 0);
        }

        void Train(float learningRate, float momentum)
        {
            BackPropagationCalculateDelta();
            BackPropagationChangeWeight(learningRate, momentum);
        }

        private void BackPropagationCalculateDelta()
        {
            //sizeNetwork.Length-2 to start at the last hidden layer
            for (int iLayer = sizeNetwork.Length - 2; iLayer >= 0; iLayer--)
            {
                for (int jNeuron = 0; jNeuron <= sizeNetwork[iLayer]; jNeuron++)
                {
                    Neuron neuron = network[iLayer, jNeuron];
                    float sumErrWeight = 0;

                    for (int kWeight = 0; kWeight < neuron.weights.Length; kWeight++)
                    {
                        sumErrWeight += network[iLayer + 1, kWeight].deltaError * neuron.weights[kWeight];
                    }
                    float deltaError = func.df(neuron.input) * sumErrWeight;

                    neuron.deltaError = deltaError;
                }
            }
        }

        protected void BackPropagationChangeWeight(float learningRate, float momentum)
        {
            //update the weights
            for (int iLayer = sizeNetwork.Length - 2; iLayer >= 0; iLayer--)
            {
                //each neuron of the hiddenLayer index i
                for (int jNeuron = 0; jNeuron <= sizeNetwork[iLayer]; jNeuron++)
                {
                    Neuron neuron = network[iLayer, jNeuron];
                    for (int kWeight = 0; kWeight < neuron.weights.Length; kWeight++)
                    {
                        float deltaW = learningRate * network[iLayer + 1, kWeight].deltaError * neuron.output;
                        neuron.weights[kWeight] += deltaW + momentum * neuron.previousDeltaError;

                        neuron.previousDeltaError = deltaW;
                    }
                }
            }
        }
    }
}