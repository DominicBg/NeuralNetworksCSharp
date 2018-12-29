using NeuralNetwork;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BrainReinforcement : Brain {

    public ReinforcementNeuralNetwork.ReinforcementInfo reinforcementInfo { get; protected set; }


    [SerializeField] float randomRatio = 0.1f;
    [SerializeField] int batchSize = 5000;
    ReinforcementNeuralNetwork reinforcementNeural;

    void Start()
    {
        neuralNetwork = new ReinforcementNeuralNetwork(sizeNetwork, batchSize);
        reinforcementNeural = (ReinforcementNeuralNetwork)neuralNetwork;

        info = neuralNetwork.info;
        reinforcementInfo = reinforcementNeural.reinforcementInfo;
    }

    public override float[] SetInputGetOutput(float[] inputs)
    {
        return null;// reinforcementNeural.SetInputGetOutput(inputs, randomRatio);
    }
    public void AddReward(float reward)
    {
        reinforcementNeural.AddReward(reward);
    }
    public void StartExperiment()
    {
        reinforcementNeural.StartExperiment();
    }
    public void EndExperiment(bool addExperienceToBuffer)
    {
        reinforcementNeural.EndExperiment(addExperienceToBuffer);
    }
}
