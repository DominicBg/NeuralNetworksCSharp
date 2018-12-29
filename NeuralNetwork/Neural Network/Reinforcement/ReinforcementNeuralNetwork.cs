using System.Collections;
using System.Collections.Generic;
using UnityEngine;

//Still in progress

namespace NeuralNetwork
{
    public class ReinforcementNeuralNetwork : BatchNeuralNetwork
    {
        List<Experience> experiences = new List<Experience>();
        public ReinforcementInfo reinforcementInfo { get; private set; }


        public ReinforcementNeuralNetwork(int[] sizeNetwork, int sampleSize) : base(sizeNetwork, sampleSize)
        {
            reinforcementInfo = new ReinforcementInfo();
        }
        public ReinforcementNeuralNetwork(int[] sizeNetwork) : base(sizeNetwork)
        {
            reinforcementInfo = new ReinforcementInfo();
        }

        public float[] SetInputGetOutput(float[] inputs, float randomRatio)
        {
            SetInput(inputs);
            Update();
            float[] outputs = GetOutputs();
            float[] randomRatios = GetRandomFactors(randomRatio);
            float[] randomisedOutput = AddOuputRandom(outputs, randomRatios);

            AddExperience(inputs, outputs, randomRatios);
            return randomisedOutput;
        }

        float[] AddOuputRandom(float[] outputs, float[] randomOuputFactors)
        {
            float[] randomisedOutput = new float[outputSize];
            for (int i = 0; i < outputSize; i++)
            {
                randomisedOutput[i] = outputs[i] + randomOuputFactors[i];
            }
            return randomisedOutput;
        }

        float[] GetRandomFactors(float randomRatio)
        {
            float halfRandom = randomRatio * 0.5f;
            float[] randomRatios = new float[outputSize];
            for (int i = 0; i < outputSize; i++)
            {
                randomRatios[i] = Random.Range(-halfRandom, halfRandom);
            }
            return randomRatios;
        }

        void AddExperience(float[] inputs, float[] outputs, float[] randomOuputFactors)
        {
            Experience exp = new Experience();
            exp.inputs = inputs;
            exp.outputs = outputs;
            exp.randomOuputFactor = randomOuputFactors;
            experiences.Add(exp);
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
            experiences.Clear();
        }
        public void EndExperiment(bool addExperienceToBuffer)
        {
            reinforcementInfo.isStarted = false;

            if(addExperienceToBuffer)
                AddToBatchTraining();

            experiences.Clear();
        }
        
        void AddToBatchTraining()
        {
            foreach(Experience exp in experiences)
            {
                float[] desiredOutput = new float[outputSize];
                for (int i = 0; i < outputSize; i++)
                {
                    //True output + reward * randomisedFactor
                    //Si le randomised etait good = reward positif, train pour atteindre le randomised value
                    //Else reward = negatif, train dans l'autre direction
                    desiredOutput[i] = exp.outputs[i] + reinforcementInfo.totalReward * exp.randomOuputFactor[i];
                }
                AddTrainingData(exp.inputs, desiredOutput);
            }
        }

        [System.Serializable]
        public class Experience
        {
            public float[] inputs;
            public float[] outputs;
            public float[] randomOuputFactor;
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