using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System.Threading;
namespace NeuralNetwork
{
    public partial class BatchNeuralNetwork : ArtificialNeuralNetwork
    {
        Thread thread;

        /// <summary>
        /// Create a thread that will call BatchTraining in a loop. 
        /// This method is optimal since it will learn in another thread
        /// Making Unity run at full capacity.
        /// 
        /// </summary>
        /// <param name="learningRate">Learning rate.</param>
        /// <param name="momentum">Momentum.</param>
        public void ThreadBatchTraining(float learningRate, float momentum)
        {
            if (!IsThreadAlive())
                StartBatchThread(learningRate, momentum);
        }

        void StartBatchThread(float learningRate, float momentum)
        {
            thread = new Thread(() => LoopBatchTraining(learningRate, momentum));
            thread.Start();
        }

        void LoopBatchTraining(float learningRate, float momentum)
        {
            while (IsThreadAlive())
                BatchTraining(learningRate, momentum);
        }

        bool IsThreadAlive()
        {
            return (thread != null && thread.ThreadState != ThreadState.Unstarted && thread.ThreadState != ThreadState.Aborted);
        }
        /// <summary>
        /// Closes the thread. Call this in void OnApplicationQuit().
        /// Otherwise Unity will freeze.
        /// </summary>
        public void CloseThread()
        {
            if (thread != null)
            {
                thread.Interrupt();
                thread.Abort();
                thread = null;
                Debug.Log("thread closed");
            }
        }
    }
}