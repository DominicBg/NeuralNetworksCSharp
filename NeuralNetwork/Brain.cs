using NeuralNetwork;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Brain : MonoBehaviour
{
    public bool isTraining;

    public BatchNeuralNetwork neuralNetwork { get; protected set; }
    public ArtificialNeuralNetwork.Info info { get; protected set; }
    public BatchNeuralNetwork.BatchInfo batchInfo { get; protected set; }

    [Header("First = input, Last = Output")]
    [SerializeField] protected int[] sizeNetwork = new int[] { 2, 3, 2 };
    [SerializeField] protected float learningRate = 0.1f;

    [Header("Advanced")]
    [SerializeField] protected float momentum = 1;
    [SerializeField] protected bool useThread = false;
    [SerializeField] Text outputLog;
    [SerializeField] string brainName;
    [SerializeField] int maxBatchSize = 50000;
    void Start()
    {
        neuralNetwork = new BatchNeuralNetwork(sizeNetwork, maxBatchSize);
        info = neuralNetwork.info;
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

        if(outputLog != null)
            outputLog.text = info.log + "\n " + info.totalDetalError;
    }

    public void Train()
    {
        if (useThread)
            neuralNetwork.ThreadBatchTraining(learningRate, momentum);
        else
            neuralNetwork.BatchTraining(learningRate, momentum);
    }

    public virtual float[] SetInputGetOutput(float[]inputs)
    {
        neuralNetwork.SetInput(inputs);
        neuralNetwork.Update();
        return neuralNetwork.GetOutputs();
    }
    public virtual float[] SetInputGetOutputRandomise(float[] inputs, float randomFactor)
    {
        neuralNetwork.SetInput(inputs);
        neuralNetwork.Update();
        return neuralNetwork.GetOutputsRandomised(randomFactor);
    }

    private void OnDestroy()
    {
        neuralNetwork.CloseThread();
    }

    [ContextMenu("Save Brain")]
    public void SaveBrain()
    {
        neuralNetwork.SaveBrain(brainName);
    }

    [ContextMenu("Load Brain")]
    public void LoadBrain()
    {
        neuralNetwork.LoadBrain(brainName);
    }
}
