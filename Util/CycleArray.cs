using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CycleArray<T>
{
    T[] arr;
    int currentIndex;
    int currentCount;
    int bufferSize;
    bool hasCycle = false;

    public CycleArray(int bufferSize)
    {
        this.bufferSize = bufferSize;
        arr = new T[bufferSize];
    }

    public void Reset()
    {
        currentIndex = 0;
        currentCount = 0;
        hasCycle = false;
    }

    public T this[int index]
    {
        get { return arr[index]; }
        //set
    }

    public void Add(T t)
    {
        arr[currentIndex] = t;

        currentIndex += 1;
        if (currentIndex == bufferSize)
        {
            currentIndex = 0;
            hasCycle = true;
        }

        currentCount += 1;
        if (currentCount > bufferSize)
        {
            currentCount = bufferSize;
        }
    }

    public T[] GetArray()
    {
        return (hasCycle) ? GetArrayCycle() : GetArrayNoCycle();
    }

    T[] GetArrayCycle()
    {
        T[] output = new T[currentCount];
        for (int i = 0; i < currentCount; i++)
        {
            int ii = (currentIndex + i) % bufferSize;
            output[i] = arr[ii];
        }
        return output;
    }

    T[] GetArrayNoCycle()
    {
        T[] output = new T[currentCount];
        for (int i = 0; i < currentCount; i++)
        {
            output[i] = arr[i];
        }
        return output;
    }
}
