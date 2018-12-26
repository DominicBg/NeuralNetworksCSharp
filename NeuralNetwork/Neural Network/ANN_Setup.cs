using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public partial class ArtificialNeuralNetwork
{
    const string ERRORLENGTH = ("<color=red>Size Network must be higher than 2</color>");
    const string ERRORINPUT = ("<color=red>Input index is out of bound</color>");
    const string ERROROUTPUT = ("<color=red>Output index is out of bound</color>");

    protected int[] sizeNetwork = new int[2];
    Neuron[,] network;
    int indexOutputLayer;

    public int[] SizeNetwork { get { return sizeNetwork; } }
    public Neuron[,] Network { get { return network; } }
    public Info info { get; private set; }

    protected int outputSize;

    TransferFunction func;

    // i = iLayer
    // j = jNeuron 
    // k = kWeight

    /// <summary>
    /// Initializes a new instance of the <see cref="NeuralNetwork"/> class.
    /// </summary>
    /// <param name="sizeNetwork">An array containing the number of neurons for each layer</param>
    public ArtificialNeuralNetwork(int[] sizeNetwork)
    {
        info = new Info();

        indexOutputLayer = sizeNetwork.Length - 1;
        outputSize = sizeNetwork[indexOutputLayer];

        InitialiseNetwork(sizeNetwork);
        SetTransferFunction(TransferFunction.Function.Sigmoid);
    }

    private void InitialiseNetwork(int[] sizeNetwork)
    {
        this.sizeNetwork = sizeNetwork;
        if (sizeNetwork.Length < 2)
            Debug.Log(ERRORLENGTH);

        //+1 for bias neuron
        int maxNeuron = MaxValueNetwork() + 1;

        network = new Neuron[sizeNetwork.Length, maxNeuron];

        for (int iLayer = 0; iLayer < sizeNetwork.Length; iLayer++)
        {
            //each layer will receive a bias unless its the output layer
            int addBias = (iLayer != indexOutputLayer) ? 1 : 0;
            for (int jNeuron = 0; jNeuron < sizeNetwork[iLayer] + addBias; jNeuron++)
            {
                if (iLayer < indexOutputLayer)
                    network[iLayer, jNeuron] = new Neuron(sizeNetwork[iLayer + 1]);
                else //dont need to set weights since its the output
                    network[iLayer, jNeuron] = new Neuron();
            }
        }
    }

    /// <summary>
    /// Change the transfer function, by default Sigmoid
    /// </summary>
    /// <param name="transferFunction"></param>
    public void SetTransferFunction(TransferFunction.Function transferFunction)
    {
        func = TransferFunction.GetTransferFunction(transferFunction);
    }

    [System.Serializable]
    public class Info
    {
        public float totalError;
        public float totalDetalError;
        public string log;
    }

    [System.Serializable]
    public class Neuron
    {
        const float randomRange = 3;

        public float input = 1;
        public float output = 1;

        public float[] weights;

        public float error; //to removeo
        public float deltaError;
        public float previousDeltaError;

        public Neuron(){}

        public Neuron(int sizeWeights)
        {
            weights = new float[sizeWeights];
            for (int i = 0; i < sizeWeights; i++)
            {
                weights[i] = RandomWeight();
            }
        }

        public static float RandomWeight()
        {
            return Random.Range(-randomRange, randomRange);
        }
    }
}
