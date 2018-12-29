using System.Collections;
using System.Collections.Generic;
using UnityEngine;
namespace NeuralNetwork
{
    public partial class ArtificialNeuralNetwork
    {
        /// <summary>
        /// Feed Forward. This will take your input and produce an output.
        /// </summary>
        public void Update()
        {
            //Starts at first hidden layer (index 1)
            for (int iLayer = 1; iLayer < sizeNetwork.Length; iLayer++)
            {
                //each neuron of the hiddenLayer index i
                for (int jNeuron = 0; jNeuron < sizeNetwork[iLayer]; jNeuron++)
                {
                    CalculateValueNeuron(iLayer, jNeuron);
                }
            }
        }


        /// <summary>
        /// Calculate the value of a given neuron
        /// [i,j] is the neuron in the network
        /// neuron[i,j] get all from neuron[i-1, j] * their weight
        /// sum it all and pass it into sigmoid function
        /// </summary>
        /// <param name="i">The index.</param>
        /// <param name="j">J.</param>
        void CalculateValueNeuron(int iLayer, int jNeuron)
        {
            Neuron neuron = network[iLayer, jNeuron];
            float output = 0;
            //+1 for bias
            for (int kWeight = 0; kWeight < sizeNetwork[iLayer - 1] + 1; kWeight++)
            {
                output += network[iLayer - 1, kWeight].output * network[iLayer - 1, kWeight].weights[jNeuron];
            }
            neuron.input = output;
            //neuron.output = Sigmoid(output);
            neuron.output = func.f(output);
        }

        /// <summary>
        /// Return the highest number of neuron for each layer
        /// </summary>
        /// <returns>Return the highest number of neuron</returns>
        int MaxValueNetwork()
        {
            int maxValue = sizeNetwork[0];
            for (int iLayer = 0; iLayer < sizeNetwork.Length; iLayer++)
            {
                maxValue = Mathf.Max(maxValue, sizeNetwork[iLayer]);
            }
            return maxValue;
        }

        /// <summary>
        /// Return the neuron at position [i,j]
        /// </summary>
        /// <returns>The neuron.</returns>
        /// <param name="iLayer">The index layer.</param>
        /// <param name="jNeuron">The index neuron.</param>
        public Neuron GetNeuron(int iLayer, int jNeuron)
        {
            return network[iLayer, jNeuron];
        }


    }
}