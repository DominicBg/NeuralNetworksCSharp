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
        TransferFunction func;
        protected int outputSize;

        public int[] SizeNetwork { get { return SizeNetwork; } }
        public Neuron[,] Network { get { return network; } }


        // i = iLayer
        // j = jNeuron 
        // k = kWeight

        public NeuralNetwork(int[] sizeNetwork)
        {
            InitialiseNetwork(sizeNetwork);
            SetTransferFunction(TransferFunction.Function.Sigmoid);
            outputSize = sizeNetwork[sizeNetwork.Length - 1];
        }

        private void InitialiseNetwork(int[] sizeNetwork)
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
        /// Change the transfer function, by default Sigmoid
        /// </summary>
        /// <param name="transferFunction"></param>
        public void SetTransferFunction(TransferFunction.Function transferFunction)
        {
            func = TransferFunction.GetTransferFunction(transferFunction);
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
            if (index > outputSize)
            {
                Debug.Log(ERROROUTPUT);
                return 0;
            }
            return network[outputSize, index].output;
        }

        public float[] GetOutputs()
        {
            float[] outputs = new float[outputSize];
            for (int i = 0; i < outputSize; i++)
            {
                outputs[i] = network[outputSize, i].output;
            }
            return outputs;
        }

        /// <summary>
        /// Gets the output of a given index.
        /// Return true if the value is over 0, else return false.
        /// </summary>
        /// <returns>The output value</returns>
        /// <param name="index">Index of the output neuron</param>
        public bool GetOutputBool(int index)
        {
            if (index > outputSize)
            {
                Debug.Log(ERROROUTPUT);
                return false;
            }
            //0 for false 1 for true, .5f is the middle point
            return (network[outputSize, index].output > .5f) ? true : false;
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
            if (index > outputSize)
            {
                Debug.Log(ERROROUTPUT);
                return 0;
            }
            return Mathf.Lerp(rangeFrom, rangeTo, network[outputSize, index].output);
        }

        /// <summary>
        /// Gets the output of a given index which will be between -1 and 1.
        /// </summary>
        /// <returns>The output value formated between -1 and 1</returns>
        /// <param name="index">Index of the output neuron</param>
        public float GetOutputNegatif(int index)
        {
            if (index > outputSize)
            {
                Debug.Log(ERROROUTPUT);
                return 0;
            }
            return 2 * network[outputSize, index].output - 1;
        }
        #endregion

        #region backpropagation
        public float[] GetDeltaError()
        {
            int iLayer = sizeNetwork.Length - 1;
            int iLayerSize = sizeNetwork[iLayer];
            float[] deltaErrors = new float[iLayerSize];
            for (int jNeuron = 0; jNeuron < iLayerSize; jNeuron++)
            {
                deltaErrors[jNeuron] = network[iLayer, jNeuron].deltaError;
            }
            return deltaErrors;
        }

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

        protected string CalculateSetError(float[] targetOutputs)
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

                float deltaError = func.df(neuron.input) * (target - output);

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
            BackPropagationCalculateError();
            BackPropagationAjustWeight(learningRate, momentum);
        }

        private void BackPropagationCalculateError()
        {
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

        protected void BackPropagationAjustWeight(float learningRate, float momentum)
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