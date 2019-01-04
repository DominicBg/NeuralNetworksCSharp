using NeuralNetwork;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
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
    [SerializeField] protected int maxBatchSize = 50000;

    public UnityEvent OnBrainTick { get; private set; }
    [SerializeField] float tickRate = 0;

    float currentTick = 0;

    void Awake()
    {
        neuralNetwork = new BatchNeuralNetwork(sizeNetwork, maxBatchSize);
        InitInfo();
    }
    protected virtual void InitInfo()
    {
        info = neuralNetwork.info;
        batchInfo = neuralNetwork.batchInfo;
        OnBrainTick = new UnityEvent();
    }

    public void AddTrainingData(float[] inputs, float[] desiredOutput, float weight = 1)
    {
        neuralNetwork.AddTrainingData(inputs, desiredOutput, weight);
    }

    void Update()
    {
        if (tickRate != 0)
        {
            currentTick -= Time.deltaTime;
            if (currentTick > 0)
                return;

            currentTick = tickRate;
            OnBrainTick.Invoke();
        }

        if (isTraining)
            Train();
        else
            neuralNetwork.CloseThread();

        ShowLog(GetLog());
    }

    protected virtual string GetLog()
    {
        return info.log +
            "\n Error = " + info.totalDetalError +
            "\n CurrentBatchSize = " + batchInfo.sampleCount;
    }

    protected void ShowLog(string log)
    {
        if (outputLog != null)
            outputLog.text = log;
    }

    public void Train()
    {
        if (useThread)
            neuralNetwork.ThreadBatchTraining(learningRate, momentum);
        else
            neuralNetwork.BatchTraining(learningRate, momentum);
    }

    public virtual ANN_Output SetInputGetOutput(float[]inputs, float randomFactor = 0)
    {
        neuralNetwork.SetInput(inputs);
        neuralNetwork.Update();
        ANN_Output output = neuralNetwork.GetANNOutput();

        if (randomFactor != 0)
        {
            output.RandomizeOutput(randomFactor);
        }

        return output;
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
