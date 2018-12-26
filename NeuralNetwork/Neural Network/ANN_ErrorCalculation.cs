using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public partial class ArtificialNeuralNetwork
{
    void CalculateSetError(float[] targetOutputs)
    {
        int iLayer = indexOutputLayer;
        if (targetOutputs.Length > sizeNetwork[iLayer])
            Debug.Log(ERRORLENGTH);

        string log = "";

        float totalError = 0;
        float totalDeltaError = 0;

        // 1/2 * sum of squared error
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
            totalDeltaError += deltaError;

            log += (jNeuron + ") ouput = " + output + "| target = " + target + "\n");

        }
        if (float.IsNaN(totalError))
            totalError = Mathf.Epsilon;
        if (float.IsNaN(totalDeltaError))
            totalDeltaError = Mathf.Epsilon;

        info.totalError = totalError;
        info.totalDetalError = totalDeltaError;
        info.log = log;
    }

}
