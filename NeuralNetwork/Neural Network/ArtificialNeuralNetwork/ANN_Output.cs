using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ANN_Output
{
    float[] output;

    public ANN_Output(float [] output)
    {
        this.output = output;
    }

    public float this[int i]
    {
        get { return output[i]; }
    }

    public static implicit operator float[](ANN_Output ann_ouput)
    {
        return ann_ouput.output;
    }

    public bool GetBool(int i)
    {
        return (output[i] > .5f) ? true : false;
    }

    public float GetBetween(int i, float from, float to)
    {
        return Mathf.Lerp(from, to, output[i]);
    }

    /// <summary>
    /// Get the value at i from -1 to 1
    /// </summary>
    /// <param name="i"></param>
    /// <returns></returns>
    public float GetAxis(int i)
    {
        return 2 * output[i] - 1;
    }

    /// <summary>
    /// Return the ouputs normalized. Every value is between 0 and 1.
    /// The sum of the value is 1.
    /// Uses the softmax functions.
    /// </summary>
    /// <returns>The outputs normalized.</returns>
    public float[] GetOutputsNormalized()
    {
        float[] expLayer = new float[output.Length];
        float sumExp = 0;

        for (int i = 0; i < output.Length; i++)
        {
            expLayer[i] = Mathf.Exp(output[i]);
            sumExp += expLayer[i];
        }

        for (int i = 0; i < output.Length; i++)
        {
            output[i] = expLayer[i] / sumExp;
        }
        return output;
    }

    /// <summary>
    /// This will normalize all input from 0 to 1 with a sum of 1.
    /// Cannot reverse this action
    /// </summary>
    public void Normalize()
    {
        output = GetOutputsNormalized();
    }

    public float MaxOutput()
    {
        return output[IndexMaxOutput()];
    }

    public int IndexMaxOutput()
    {
        float max = int.MinValue;
        int maxIndex = 0;
        for (int i = 0; i < output.Length; i++)
        {
            if (output[i] > max)
            {
                maxIndex = i;
                max = output[i];
            }
        }
        return maxIndex;
    }

    /// <summary>
    /// This will randomize all output, cannot reverse this action.
    /// </summary>
    /// <param name="randomFactor"></param>
    public void RandomizeOutput(float randomFactor)
    {
        float halfRandomFactor = randomFactor * 0.5f;
        for (int i = 0; i < output.Length; i++)
        {
            output[i] += Random.Range(-halfRandomFactor, halfRandomFactor);
        }
    }
}
