using NeuralNetwork;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BrainReinforcement : Brain {

    public ReinforcementNeuralNetwork.ReinforcementInfo reinforcementInfo { get; protected set; }

    [SerializeField] float randomRatio = 0.1f;
    [SerializeField] int experienceBufferSize = 25;
    ReinforcementNeuralNetwork reinforcementNeural;

    void Start()
    {
        neuralNetwork = new ReinforcementNeuralNetwork(sizeNetwork, maxBatchSize, experienceBufferSize);
        reinforcementNeural = (ReinforcementNeuralNetwork)neuralNetwork;

        InitInfo();
    }

    protected override void InitInfo()
    {
        base.InitInfo();
        reinforcementInfo = reinforcementNeural.reinforcementInfo;
    }

    public override ANN_Output SetInputGetOutput(float[] inputs, float randomRatio = 1)
    {
        ANN_Output output = base.SetInputGetOutput(inputs, randomRatio);

        if(reinforcementInfo.isStarted)
            reinforcementNeural.AddExperience(inputs, output);

        return output;
    }
    public void AddReward(float reward)
    {
        reinforcementNeural.AddReward(reward);
    }
    public void StartExperiment()
    {
        reinforcementNeural.StartExperiment();
    }
    public void EndExperiment()
    {
        reinforcementNeural.EndExperiment();
    }

    protected override string GetLog()
    {
        string brainLog = base.GetLog();
        brainLog += 
            "\n Is started = " + reinforcementInfo.isStarted +
            "\n total reward = " + reinforcementInfo.totalReward +
            "\n exp count = " + reinforcementInfo.experienceCount;

        return brainLog;
    }

}
