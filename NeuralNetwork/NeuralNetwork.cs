using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using UnityEditor;

/// <summary>
/// Author : Dominic Brodeur-Gendron
/// </summary>
namespace NeuralNetwork
{

    [System.Serializable]
    public class NeuralNetwork
    {
        const string ERRORLENGTH = ("<color=red>Size Network must be higher than 2</color>");
        const string ERRORINPUT = ("<color=red>Input index is out of bound</color>");
        const string ERROROUTPUT = ("<color=red>Output index is out of bound</color>");

        int[] sizeNetwork = new int[2];
        Neuron[,] network;

        // i = iLayer
        // j = jNeuron 
        // k = kWeight

        public NeuralNetwork(int[] sizeNetwork)
        {
            this.sizeNetwork = sizeNetwork;
            if (sizeNetwork.Length < 2)
                Debug.Log(ERRORLENGTH);

            //+1 for bias neuron
            int maxNeuron = MaxValueNetwork() + 1;
            network = new Neuron[sizeNetwork.Length, maxNeuron];

            for (int iLayer = 0; iLayer < sizeNetwork.Length; iLayer++)
            {
                //each layer will receive a bias unless its the output layer
                int addBias = (iLayer != sizeNetwork.Length - 1) ? 1 : 0;
                for (int jNeuron = 0; jNeuron < sizeNetwork[iLayer] + addBias; jNeuron++)
                {
                    if (iLayer < sizeNetwork.Length - 1)
                        network[iLayer, jNeuron] = new Neuron(sizeNetwork[iLayer + 1]);
                    else //dont need to set weights since its the output
                        network[iLayer, jNeuron] = new Neuron();
                }
            }
        }

        /// <summary>
        /// Feed Forward 
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
            neuron.output = Sigmoid(output);
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

        public Neuron GetNeuron(int iLayer, int jNeuron)
        {
            return network[iLayer, jNeuron];
        }

        #region I/O
        /// <summary>
        /// Sets the input at a given index to a given value.
        /// </summary>
        /// <param name="index">Index of the input neuron</param>
        /// <param name="value">The Value.</param>
        public void SetInput(int index, float value)
        {
            if (index > sizeNetwork[0])
            {
                Debug.Log(ERRORINPUT);
                return;
            }

            network[0, index].output = value;
        }

        public void SetInput(float[] values)
        {
            if (values.Length > sizeNetwork[0])
            {
                Debug.Log(ERRORINPUT);
                return;
            }

            for (int i = 0; i < values.Length; i++)
                network[0, i].output = values[i];
        }

        /// <summary>
        /// Gets the output of a given index.
        /// </summary>
        /// <returns>The output value</returns>
        /// <param name="index">Index of the output neuron</param>
        public float GetOutput(int index)
        {
            if (index > sizeNetwork[sizeNetwork.Length - 1])
            {
                Debug.Log(ERROROUTPUT);
                return 0;
            }
            return network[sizeNetwork.Length - 1, index].output;
        }

        /// <summary>
        /// Gets the output of a given index.
        /// Return true if the value is over 0, else return false.
        /// </summary>
        /// <returns>The output value</returns>
        /// <param name="index">Index of the output neuron</param>
        public bool GetOutputBool(int index)
        {
            if (index > sizeNetwork[sizeNetwork.Length - 1])
            {
                Debug.Log(ERROROUTPUT);
                return false;
            }
            //0 for false 1 for true, .5f is the middle point
            return (network[sizeNetwork.Length - 1, index].output > .5f) ? true : false;
        }

        /// <summary>
        /// Gets the output of a given index which will be between rangefrom and rangeTo.
        /// Exemple : GetOutputRange(1,-1,1)
        /// Return the output formated between -1 and 1;
        /// </summary>
        /// <returns>The output value formated between from and to</returns>
        /// <param name="index">Index of the output neuron</param>
        /// <param name="rangeFrom">The lower limit of the output</param>
        /// <param name="rangeTo">The upper limit of the output</param>
        public float GetOutputRange(int index, float rangeFrom, float rangeTo)
        {
            if (index > sizeNetwork[sizeNetwork.Length - 1])
            {
                Debug.Log(ERROROUTPUT);
                return 0;
            }
            return Mathf.Lerp(rangeFrom, rangeTo, network[sizeNetwork.Length - 1, index].output);
        }

        /// <summary>
        /// Gets the output of a given index which will be between -1 and 1.
        /// </summary>
        /// <returns>The output value formated between -1 and 1</returns>
        /// <param name="index">Index of the output neuron</param>
        public float GetOutputNegatif(int index)
        {
            if (index > sizeNetwork[sizeNetwork.Length - 1])
            {
                Debug.Log(ERROROUTPUT);
                return 0;
            }
            return 2 * network[sizeNetwork.Length - 1, index].output - 1;
        }
        #endregion

        #region save/load
        public void SaveBrain()
        {
            SaveBrain("");
        }
        public void SaveBrain(string additional)
        {
            string str = "";
            for (int i = 0; i < sizeNetwork.Length - 1; i++)
            {
                //+1 for bias
                for (int j = 0; j < sizeNetwork[i] + 1; j++)
                {
                    if (network[i, j] == null)
                        continue;

                    for (int k = 0; k < network[i, j].weights.Length; k++)
                    {
                        str += network[i, j].weights[k];
                        if (k != network[i, j].weights.Length - 1)
                            str += " ";
                    }

                    str += "|" + network[i, j].deltaError + "|\n"; //each neuron
                }
                str += "\n"; //each layer
            }
            string path = "Assets/Resources/NeuralNetworkValue" + additional + ".txt";
            StreamWriter writer = new StreamWriter(path, false);
            writer.WriteLine(str);
            writer.Close();
        }
        public void LoadBrain()
        {
            LoadBrain("");
        }
        public void LoadBrain(string additional)
        {
            string path = "Assets/Resources/NeuralNetworkValue" + additional + ".txt";
            StreamReader reader = new StreamReader(path);

            for (int i = 0; i < sizeNetwork.Length - 1; i++)
            {
                //+1 for bias
                for (int j = 0; j < sizeNetwork[i] + 1; j++)
                {
                    string[] strWeights = reader.ReadLine().Split(' ');
                    float[] weights = new float[strWeights.Length];
                    for (int k = 0; k < strWeights.Length; k++)
                    {
                        Debug.Log(strWeights[k]);
                        weights[k] = System.Convert.ToSingle(strWeights[k]);
                    }
                    network[i, j].weights = weights;
                }
            }
            reader.Close();
        }
        #endregion

        #region static
        /// <summary>
        /// Mixs the weights of 2 Neural Networks with a chance (mutationRate) to randomise a weight.
        /// </summary>
        /// <returns>The newest Neural Network</returns>
        /// <param name="papa">First Neural Network.</param>
        /// <param name="maman">Second Neural Network.</param>
        /// <param name="sizeNetwork">Size network.</param>
        /// <param name="mutationRate">Mutation rate (Between 0 and 1).</param>
        public static NeuralNetwork MixNeuron(NeuralNetwork papa, NeuralNetwork maman, int[] sizeNetwork, float mutationRate)
        {
            NeuralNetwork baby = new NeuralNetwork(sizeNetwork);

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
        #endregion

        #region backpropagation

        //Maybe to remove
        public void SetDeltaError(float[] deltaErrors)
        {
            int iLayer = sizeNetwork.Length - 1;
            if (deltaErrors.Length > sizeNetwork[iLayer])
                Debug.Log(ERRORLENGTH);

            for (int jNeuron = 0; jNeuron < deltaErrors.Length; jNeuron++)
            {
                network[iLayer, jNeuron].deltaError = deltaErrors[jNeuron];
            }
        }

        //Maybe to remove
        public float[] GetDeltaError(float[] targetOutputs)
        {
            string str = "";
            return GetDeltaError(targetOutputs, ref str);
        }
        //Maybe to remove
        public float[] GetDeltaError(float[] targetOutputs, ref string log)
        {
            int iLayer = sizeNetwork.Length - 1;
            if (targetOutputs.Length > sizeNetwork[iLayer])
                Debug.Log(ERRORLENGTH);

            float[] deltaErrors = new float[targetOutputs.Length];
            float totalError = 0;
            log = "";
            for (int jNeuron = 0; jNeuron < targetOutputs.Length; jNeuron++)
            {
                Neuron neuron = network[iLayer, jNeuron];
                float output = GetOutput(jNeuron);
                float target = targetOutputs[jNeuron];

                float error = target - output;
                error = (error * error) / 2;


                float deltaError = Sigmoid_d1(neuron.input) * (target - output);
                deltaErrors[jNeuron] = deltaError;

                log += (jNeuron + ") ouput = " + output + "| target = " + target + "\n");

                totalError += error;

            }
            log += "Total Error = " + totalError.ToString("N4") + "\n";
            return deltaErrors;
        }

        private string CalculateSetError(float[] targetOutputs)
        {
            int iLayer = sizeNetwork.Length - 1;
            if (targetOutputs.Length > sizeNetwork[iLayer])
                Debug.Log(ERRORLENGTH);

            string log = "";

            // 1/2 * sum of squared error
            float totalError = 0;
            for (int jNeuron = 0; jNeuron < targetOutputs.Length; jNeuron++)
            {
                Neuron neuron = network[iLayer, jNeuron];
                float output = GetOutput(jNeuron);
                float target = targetOutputs[jNeuron];


                float error = target - output;
                error = (error * error) / 2;

                float deltaError = Sigmoid_d1(neuron.input) * (target - output);

                network[iLayer, jNeuron].error = error;
                network[iLayer, jNeuron].deltaError = deltaError;
                totalError += error;
                log += (jNeuron + ") ouput = " + output + "| target = " + target + "\n");

            }
            if (float.IsNaN(totalError))
                totalError = Mathf.Epsilon;

            log += "Total Error" + totalError + "\n";
            return log;
        }

        public string TrainWithExemples(float[] targetOutputs, float learningRate)
        {
            return TrainWithExemples(targetOutputs, learningRate, 0);
        }

        public string TrainWithExemples(float[] targetOutputs, float learningRate, float momentum)
        {
            if (targetOutputs.Length > sizeNetwork[sizeNetwork.Length - 1])
                Debug.Log(ERRORLENGTH);

            string log = CalculateSetError(targetOutputs);
            Train(learningRate, momentum);
            return log;
        }

        /// <summary>
        /// Make the Neural Network learn from exemples.
        /// </summary>
        /// <param name="targetOutputs">Target outputs/Exemples.</param>
        /// <param name="learningRate">Learning rate. Around 0.1f</param>
        /// <returns>The learning log (Optional)</returns>
        public void Train(float learningRate)
        {
            Train(learningRate, 0);
        }

        /// <summary>
        /// Make the Neural Network learn from exemples.
        /// </summary>
        /// <param name="targetOutputs">Target outputs/Exemples.</param>
        /// <param name="learningRate">Learning rate. Around 0.1f</param>
        /// <param name="momentum">Momentun. Around 0.1f</param>
        /// <returns>The learning log (Optional)</returns>
        public void Train(float learningRate, float momentum)
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
                    float deltaError = Sigmoid_d1(neuron.input) * sumErrWeight;
                    neuron.deltaError = deltaError;
                }
            }

            BackPropagation(learningRate, momentum);
        }

        protected void BackPropagation(float learningRate, float momentum)
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
        #endregion

        #region transferFunction

        public float Sigmoid(float x)
        {
            return 1 / (1 + Mathf.Exp(-x));
        }
        public float Sigmoid_d1(float x)
        {
            float s = Sigmoid(x);
            float y = s * (1 - s);
            if (y != 0)
                return y;
            else
            {
                //Debug.Log(0);
                return 0; //Mathf.Epsilon;
            }
        }

        /// <summary>
        /// Normalizeds the sigmoid from 0 > 1, to  -1 > 1.
        /// </summary>
        /// <returns>The sigmoid.</returns>
        /// <param name="x">The x coordinate.</param>
        float NormalizedSigmoid(float x)
        {
            return 2 * (Sigmoid(x)) - 1;
        }
        #endregion
    }

    [System.Serializable]
    public class Neuron
    {
        const float randomRange = 3;

        public float input = 1;
        public float output = 1;

        public float[] weights;

        public float error; //to remove
        public float deltaError;
        public float previousDeltaError;

        public Neuron(){}

        public Neuron(int sizeWeights)
        {
            weights = new float[sizeWeights];
            for (int i = 0; i < sizeWeights; i++)
            {
                weights[i] = RandomWeight();
            }
        }

        public static float RandomWeight()
        {
            return Random.Range(-randomRange, randomRange);
        }
    }
}