using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public abstract class TransferFunction
{
    public enum Function { Sigmoid, ReLU};
    public static TransferFunction GetTransferFunction(Function func)
    {
        switch(func)
        {
            case Function.Sigmoid:
                return new TransferFunctionSigmoid();
            case Function.ReLU:
                return new TransferFunctionReLU();
        }
        return null;
    }

    public abstract float f(float x);
    public abstract float df(float x);
    public abstract float normalizedF(float x);
}
