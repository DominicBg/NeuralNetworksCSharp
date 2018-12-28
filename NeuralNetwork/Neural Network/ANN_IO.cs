using System.Collections;
using System.Collections.Generic;
using UnityEngine;
namespace NeuralNetwork
{
    public partial class ArtificialNeuralNetwork
    {
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

        /// <summary>
        /// Sets the inputs
        /// </summary>
        /// <param name="values">Values.</param>
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
            return network[indexOutputLayer, index].output;
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
            return (network[indexOutputLayer, index].output > .5f) ? true : false;
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
            return Mathf.Lerp(rangeFrom, rangeTo, network[indexOutputLayer, index].output);
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
            return 2 * network[indexOutputLayer, index].output - 1;
        }

        /// <summary>
        /// Return the ouputs normalized. Every value is between 0 and 1.
        /// The sum of the value is 1.
        /// Uses the softmax functions.
        /// </summary>
        /// <returns>The outputs normalized.</returns>
        public float[] GetOutputsNormalized()
        {
            float[] outputLayer = new float[outputSize];
            for (int i = 0; i < outputLayer.Length; i++)
                outputLayer[i] = network[indexOutputLayer, i].output;

            float[] expLayer = new float[outputLayer.Length];
            float sumExp = 0;

            for (int i = 0; i < outputLayer.Length; i++)
            {
                expLayer[i] = Mathf.Exp(outputLayer[i]);
                sumExp += expLayer[i];
            }

            for (int i = 0; i < outputLayer.Length; i++)
            {
                outputLayer[i] = expLayer[i] / sumExp;
            }
            return outputLayer;
        }

        public float[] GetOutputs()
        {
            float[] outputs = new float[outputSize];
            for (int i = 0; i < outputSize; i++)
            {
                outputs[i] = network[indexOutputLayer, i].output;
            }
            return outputs;
        }

        public float[] GetOutputsRandomised(float randomFactor)
        {
            float[] outputs = GetOutputs();
            float halfRandomFactor = randomFactor * 0.5f;
            for (int i = 0; i < outputSize; i++)
            {
                outputs[i] += Random.Range(-halfRandomFactor, halfRandomFactor); ;
            }
            return outputs;
        }
    }
}