using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace NeuralNetwork
{ 
    public class ANNTester : MonoBehaviour
    {

	    NeuralNetwork neuralNetwork;
	    [SerializeField]int[] sizeNetwork;
	    [SerializeField]float[] inputs;
	    [SerializeField]float[] outputs;
	    [SerializeField]int loopPerFrame;
	    [SerializeField]float learningRate;

	    public Text outputLog;
	    void Start()
	    {
		    neuralNetwork = new NeuralNetwork(sizeNetwork);
	    }

	    void Update () 
	    {
		    for (int i = 0; i < inputs.Length; i++)
		    {
			    neuralNetwork.SetInput (i,inputs[i]);
		    }
		

		    for (int i = 0; i < loopPerFrame; i++)
		    {
			    neuralNetwork.Update();
		    }

	    }

	    [ContextMenu("save")]
	    public void SaveTop1NeuralNetwork()
	    {
		    neuralNetwork.SaveBrain("test");
	    }
    }
}