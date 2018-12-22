﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace NeuralNetwork
{ 
    public class ANNTester : MonoBehaviour
    {

	    NeuralNetwork neuralNetwork;
	    [SerializeField]int[] sizeNetwork;
	    [SerializeField] float learningRate;
        [SerializeField] float momentum;
        [SerializeField] bool isTraining;
        [SerializeField] int trainPerFrame;
        [SerializeField] Exemple exemple;
	    public Text outputLog;

        [Header("debug")]
        [SerializeField] float[] deltaErrors;
        [SerializeField] float[] inputs;
        [SerializeField] float[] ouputs;

        void Start()
	    {
		    neuralNetwork = new NeuralNetwork(sizeNetwork);
            neuralNetwork.SetTransferFunction(TransferFunction.Function.Sigmoid);
	    }

	    void Update ()
        {
            if (isTraining)
            {
                for (int i = 0; i < trainPerFrame; i++)
                {
                    Train();

                }
            }
            if (Input.GetKeyDown(KeyCode.Space))
                Use();
        }

        private void Train()
        {
            neuralNetwork.SetInput(exemple.inputs);
            neuralNetwork.Update();
            neuralNetwork.TrainWithExemples(exemple.outputs, learningRate, momentum);
            deltaErrors = neuralNetwork.GetDeltaError();
            ouputs = neuralNetwork.GetOutputs();    
        }

        private void Use()
        {
            neuralNetwork.SetInput(exemple.inputs);
            neuralNetwork.Update();
            deltaErrors = neuralNetwork.GetDeltaError();
            ouputs = neuralNetwork.GetOutputs();
        }

        [ContextMenu("save")]
	    public void SaveTop1NeuralNetwork()
	    {
		    neuralNetwork.SaveBrain("test");
	    }
    }
}