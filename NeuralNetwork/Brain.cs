using NeuralNetwork;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Brain : MonoBehaviour
{
    public bool isTraining;

    public ThreadNeuralNetwork neuralNetwork { get; private set; }
    [Header("First = input, Last = Output")]
    [SerializeField] int[] sizeNetwork = new int[] { 2, 3, 2 };
    [SerializeField] float learningRate = 0.1f;

    [Header("Advanced")]
    [SerializeField] float momentum = 1;
    [SerializeField] bool useThread = false;

    void Start()
    {
        neuralNetwork = new ThreadNeuralNetwork(sizeNetwork);
    }

    public void AddTrainingData(float[] inputs, float[] desiredOutput)
    {
        neuralNetwork.AddTrainingData(inputs, desiredOutput);
    }

    void Update()
    {
        if (isTraining)
            Train();
        else
            neuralNetwork.CloseThread();
    }

    public void Train()
    {
        if (useThread)
            neuralNetwork.ThreadBatchTraining(learningRate, momentum);
        else
            neuralNetwork.BatchTraining(learningRate, momentum);
    }

    public float[] SetInputGetOutput(float[]inputs)
    {
        neuralNetwork.SetInput(inputs);
        neuralNetwork.Update();
        return neuralNetwork.GetOutputs();
    }

    private void OnDestroy()
    {
        neuralNetwork.CloseThread();
    }
}
