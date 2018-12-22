using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TransferFunctionSigmoid : TransferFunction
{
    public override float df(float x)
    {
        float s = f(x);
        float y = s * (1 - s);
        if (y != 0)
            return y;
        else
        {
            //return 0;
           return Mathf.Epsilon;
        }
    }

    public override float f(float x)
    {
        return 1 / (1 + Mathf.Exp(-x));
    }

    public override float normalizedF(float x)
    {
        return 2 * (f(x)) - 1;
    }
}
