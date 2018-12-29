using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TransferFunctionReLU : TransferFunction
{
    public override float df(float x)
    {
        if (x > 0)
            return 1;
        else if (x < 0)
            return 0;
        else
            return 0; //undefined
    }

    public override float f(float x)
    {
        return Mathf.Max(0, x);
    }

    public override float normalizedF(float x)
    {
        return Mathf.Max(0, Mathf.Min(1,x));
    }
}
