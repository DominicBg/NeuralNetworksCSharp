using System.Collections;
using System.Collections.Generic;
using UnityEngine;

//Still in progress

namespace NeuralNetwork
{
    public class ReinforcementNeuralNetwork : BatchNeuralNetwork
    {
        const int DEFAULTBUFFERSIZE = 75;

        CircularArray<Experience> experienceBuffer;

        public ReinforcementInfo reinforcementInfo { get; private set; }

        public ReinforcementNeuralNetwork(int[] sizeNetwork, int sampleSize, int experienceBufferSize = DEFAULTBUFFERSIZE) : base(sizeNetwork, sampleSize)
        {
            reinforcementInfo = new ReinforcementInfo();
            experienceBuffer = new CircularArray<Experience>(experienceBufferSize);
        }
        public ReinforcementNeuralNetwork(int[] sizeNetwork, int experienceBufferSize = DEFAULTBUFFERSIZE) : base(sizeNetwork)
        {
            reinforcementInfo = new ReinforcementInfo();
            experienceBuffer = new CircularArray<Experience>(experienceBufferSize);
        }

        public void AddExperience(float[] inputs, float[] outputs)
        {
            Experience exp = new Experience();
            exp.inputs = inputs;
            exp.outputs = outputs;

            experienceBuffer.Add(exp);
        }

        public void AddReward(float reward)
        {
            if (!reinforcementInfo.isStarted)
            {
                Debug.LogError("Not started");
                return;
            }

            reinforcementInfo.totalReward += reward;
        }

        public void StartExperiment()
        {
            reinforcementInfo.totalReward = 0;

            if (!reinforcementInfo.isStarted)
                reinforcementInfo.experienceCount++;

            reinforcementInfo.isStarted = true;
          
            experienceBuffer.Reset();
        }

        public void EndExperiment()
        {
            reinforcementInfo.isStarted = false;

            if(reinforcementInfo.totalReward != 0)
                AddToBatchTraining();
        }

        void AddToBatchTraining()
        {
            //Calculate weight factor
            float weightFactor = (1 / experienceBuffer.Length) * reinforcementInfo.totalReward; 

            Experience[] experiences = experienceBuffer.GetArray();
            for (int i = 0; i < experiences.Length; i++)
            {
                //lastest experiences are more important
                float weight = weightFactor * (i + 1);
                Experience exp = experiences[i];
                AddTrainingData(exp.inputs, exp.outputs, weight);
            }
        }

        [System.Serializable]
        public class Experience
        {
            public float[] inputs;
            public float[] outputs;
        }

        [System.Serializable]
        public class ReinforcementInfo
        {
            public bool isStarted;
            public float totalReward;
            public float experienceCount;
        }
    }
}