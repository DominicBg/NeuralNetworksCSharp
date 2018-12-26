using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
namespace NeuralNetwork
{
    public static class NeuralNetworkExtension
    {
        public static void SaveBrain(this ArtificialNeuralNetwork nn)
        {
            SaveBrain(nn, "");
        }
        public static void SaveBrain(this ArtificialNeuralNetwork nn, string additional)
        {
            string str = "";
            for (int i = 0; i < nn.SizeNetwork.Length - 1; i++)
            {
                //+1 for bias
                for (int j = 0; j < nn.SizeNetwork[i] + 1; j++)
                {
                    if (nn.Network[i, j] == null)
                        continue;

                    for (int k = 0; k < nn.Network[i, j].weights.Length; k++)
                    {
                        str += nn.Network[i, j].weights[k];
                        if (k != nn.Network[i, j].weights.Length - 1)
                            str += " ";
                    }

                    str += "|" + nn.Network[i, j].deltaError + "|\n"; //each neuron
                }
                str += "\n"; //each layer
            }
            string path = "Assets/Resources/NeuralNetworkValue" + additional + ".txt";
            StreamWriter writer = new StreamWriter(path, false);
            writer.WriteLine(str);
            writer.Close();
        }
        public static void LoadBrain(this ArtificialNeuralNetwork nn)
        {
            LoadBrain(nn, "");
        }
        public static void LoadBrain(this ArtificialNeuralNetwork nn, string additional)
        {
            string path = "Assets/Resources/NeuralNetworkValue" + additional + ".txt";
            StreamReader reader = new StreamReader(path);

            for (int i = 0; i < nn.SizeNetwork.Length - 1; i++)
            {
                //+1 for bias
                for (int j = 0; j < nn.SizeNetwork[i] + 1; j++)
                {
                    string[] strWeights = reader.ReadLine().Split(' ');
                    float[] weights = new float[strWeights.Length];
                    for (int k = 0; k < strWeights.Length; k++)
                    {
                        Debug.Log(strWeights[k]);
                        weights[k] = System.Convert.ToSingle(strWeights[k]);
                    }
                    nn.Network[i, j].weights = weights;
                }
            }
            reader.Close();
        }

        /// <summary>
        /// Mixs the weights of 2 Neural Networks with a chance (mutationRate) to randomise a weight.
        /// Usefull when combined with Genetic Algorithms
        /// </summary>
        /// <returns>The newest Neural Network</returns>
        /// <param name="papa">First Neural Network.</param>
        /// <param name="maman">Second Neural Network.</param>
        /// <param name="sizeNetwork">Size network.</param>
        /// <param name="mutationRate">Mutation rate (Between 0 and 1).</param>
        public static ArtificialNeuralNetwork MixNeuron(ArtificialNeuralNetwork papa, ArtificialNeuralNetwork maman, int[] sizeNetwork, float mutationRate)
        {
            ArtificialNeuralNetwork baby = new ArtificialNeuralNetwork(sizeNetwork);

            //recopy every neuron up to the output layer
            for (int iLayer = 0; iLayer < sizeNetwork.Length - 1; iLayer++)
            {
                //each neuron in the layer i, +1 for bias
                for (int jNeuron = 0; jNeuron < sizeNetwork[iLayer] + 1; jNeuron++)
                {
                    //each weight in the current Neuron
                    for (int kWeight = 0; kWeight < papa.GetNeuron(iLayer, jNeuron).weights.Length; kWeight++)
                    {
                        float weight = 0;
                        if ((float)Random.Range(0f, 1f) < mutationRate)
                        {
                            weight = Neuron.RandomWeight();
                        }
                        else
                        {
                            if (Random.Range(0, 2) == 0)
                            {
                                weight = papa.GetNeuron(iLayer, jNeuron).weights[kWeight];
                            }
                            else
                            {
                                weight = maman.GetNeuron(iLayer, jNeuron).weights[kWeight];
                            }
                        }
                        baby.GetNeuron(iLayer, jNeuron).weights[kWeight] = weight;
                    }
                }
            }
            return baby;
        }
    }
}