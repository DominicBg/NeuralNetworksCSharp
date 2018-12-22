using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HowToUseTheBrain : MonoBehaviour
{
	void HowToUse()
    {
         Brain brain = GetComponent<Brain>();

        //Gather exemples input with the desired output
        //Here is the XOR exemple
        float[] input1 = new float[] { 0, 0 };
        float[] desiredOutput1 = new float[] { 0 };

        float[] input2 = new float[] { 1, 1 };
        float[] desiredOutput2 = new float[] { 0 };

        float[] input3 = new float[] { 0, 1 };
        float[] desiredOutput3 = new float[] { 1 };

        float[] input4 = new float[] { 1, 0 };
        float[] desiredOutput4 = new float[] { 1 };

        //Add exemples
        brain.AddTrainingData(input1, desiredOutput1);
        brain.AddTrainingData(input2, desiredOutput2);
        brain.AddTrainingData(input3, desiredOutput3);
        brain.AddTrainingData(input4, desiredOutput4);

        //Train the neural network on demand
        brain.Train();

        //Or set the training on automatic
        brain.isTraining = true;

        //---Once its trained---//

        //Get Values somewhere
        float[] perceptions = new float[] { 1, 1 };

        //Receive actions from brain
        float[] actions = brain.SetInputGetOutput(perceptions);

    }
}
