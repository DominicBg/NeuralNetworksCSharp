using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace NeuralNetwork
{ 
    public class ANNBatchTester : MonoBehaviour
    {

	    BatchNeuralNetwork neuralNetwork;
	    [SerializeField]int[] sizeNetwork;

        [SerializeField] Exemple[] exemples;

	    [SerializeField] int numEpoch;
	    [SerializeField] float learningRate;
        [SerializeField] float momentum;
        [SerializeField] bool isTraining;
	    public Text outputLog;

        [Header("debug")]
        [SerializeField] float[] deltaErrors;
        [SerializeField] float[] inputs;
        [SerializeField] float[] ouputs;

        void Start()
	    {
		    neuralNetwork = new BatchNeuralNetwork(sizeNetwork);
	    }

	    void Update ()
        {
            if (isTraining)
                Train();
            

            if(Input.GetKeyDown(KeyCode.Space))
                Use();

        }

        private void Train()
        {
            foreach (Exemple exemple in exemples)
            {
                neuralNetwork.SetInput(exemple.inputs);
                neuralNetwork.AddTrainingData(neuralNetwork.GetOutputs(), exemple.outputs);
            }

            neuralNetwork.Batch(learningRate, momentum, numEpoch);
            deltaErrors = neuralNetwork.GetDeltaError();
        }

        private void Use()
        {
            neuralNetwork.SetInput(inputs);
            neuralNetwork.Update();
            ouputs = neuralNetwork.GetOutputs();
        }

        [ContextMenu("save")]
	    public void SaveTop1NeuralNetwork()
	    {
		    neuralNetwork.SaveBrain("test");
	    }
    }
    [System.Serializable]
    public class Exemple
    {
        public float[] inputs;
        public float[] outputs;
    }
}