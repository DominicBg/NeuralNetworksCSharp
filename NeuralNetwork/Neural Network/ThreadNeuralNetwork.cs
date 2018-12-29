//using System.Collections;
//using System.Collections.Generic;
//using UnityEngine;

///// <summary>
///// Author : Dominic Brodeur-Gendron
///// 
///// /// To train #1 ///
///// Use AddTrainingData(data);
///// To add exemples.
///// Call BatchTraining();
///// To train based on multiple exemples.
///// 
///// For peak learning, call BatchTraining()
///// in a loop each frame, and only add good exemples.
///// 
///// Or
///// 
///// /// To train #2 ///
///// Use AddTrainingData(data);
///// To add exemples.
///// Call ThreadBatchTraining(); in update
///// This will create a new thread when the thread is aborted.
///// This will run at max capacity withou slowing down Unity.
///// (You must call CloseThread() at the end of your program
///// or Unity will crash)
///// 
///// /// /// To use ///
///// Use SetInputs(data);
///// Use Update();
///// Use GetOuput(); 
///// to get the results
///// 
///// And voila!
///// 
///// </summary>
//using System.Threading;
//namespace NeuralNetwork
//{
//    [System.Serializable]
//    public class ThreadNeuralNetwork : ArtificialNeuralNetwork
//    {

//        public BatchInfo batchInfo { get; private set; }

//        //ERRORS
//        const string ERRORLENGTH = ("<color=red>Output and desired ouput must be the same size</color>");

//        //Batch data
//        float[][] batchInputs;
//        float[][] batchDesiredOutput;
//        int[] randomIndices;

//        //Sizes
//        const int BASEBATCHSIZE = 50;
//        int batchSize;
//        int sampleInputSize;
//        int sampleOuputSize;

//        //Cycle through exemples
//        int trainingCycleCount = 0;
//        int sampleCycleCount = 0;

//        int sampleCount = 0;

        
//        public bool isCycleLoop = true;

//        public ThreadNeuralNetwork(int[] sizeNetwork) : this(sizeNetwork, BASEBATCHSIZE) {}

//        public ThreadNeuralNetwork(int[] sizeNetwork, int batchSize) : base(sizeNetwork)
//        {
//            batchInfo = new BatchInfo();

//            this.batchSize = batchSize;
//            //Error sample can only be the size of the output layer
//            sampleOuputSize = sizeNetwork[sizeNetwork.Length - 1];
//            sampleInputSize = sizeNetwork[0];

//            batchInputs = new float[batchSize][];
//            batchDesiredOutput = new float[batchSize][];

//            for (int i = 0; i < batchSize; i++)
//            {
//                batchInputs[i] = new float[sampleInputSize];
//                batchDesiredOutput[i] = new float[sampleOuputSize];
//            }
//        }

//        /// <summary>
//        /// Add one training data to the batch.
//        /// Adding more training data will give more exemple.
//        /// </summary>
//        /// <param name="inputs">The current inputs.</param>
//        /// <param name="desiredOutput">The desired output.</param>
//        public void AddTrainingData(float[] inputs, float[] desiredOutput)
//        {
//            if (inputs.Length != SizeNetwork[0] || desiredOutput.Length != outputSize)
//            {
//                Debug.LogError(ERRORLENGTH);
//                Debug.LogError("Current input size = " + inputs.Length + ", size network input = " + SizeNetwork[0]);
//                Debug.LogError("Desired output size = " + desiredOutput.Length + ", size network output = " + outputSize);

//                return;
//            }
//            if (!isCycleLoop && sampleCount == batchSize)
//            {
//                //Got all the samples
//                //will learn over those samples
//                return;
//            }

//            for (int i = 0; i < inputs.Length; i++)
//                batchInputs[sampleCycleCount][i] = inputs[i];

//            for (int i = 0; i < desiredOutput.Length; i++)
//                batchDesiredOutput[sampleCycleCount][i] = desiredOutput[i];

//            sampleCycleCount = (sampleCycleCount + 1) % batchSize;
//            sampleCount = Mathf.Min(sampleCount + 1, batchSize);
//            batchInfo.sampleCount = sampleCount;
//            UpdateRandomIndices();
//        }

//        void UpdateRandomIndices()
//        {
//            randomIndices = new int[sampleCount];
//            for (int i = 0; i < sampleCount; i++)
//            {
//                randomIndices[i] = i;
//            }
//            randomIndices.Shuffle();
//        }

//        /// <summary>
//        /// <para>
//        /// Train the neural network using all the exemples
//        /// previously added with the method AddTrainingData.
//        /// </para>
//        /// <para>
//        /// For best results, call this method, 
//        /// then set the new inputs and update the neural network.
//        /// </para>
//        /// 
//        /// </summary>
//        /// <param name="learningRate">Learning rate.</param>
//        /// <param name="momentum">Momentum.</param>
//        public void BatchTraining(float learningRate, float momentum)
//        {
//            if (sampleCount == 0)
//                return;

//            int index = randomIndices[trainingCycleCount];
//            SetInput(batchInputs[index]);
//            Update();
//            TrainWithExemples(batchDesiredOutput[index], learningRate, momentum);

//            trainingCycleCount++;
//            if(trainingCycleCount == sampleCount)
//            {
//                trainingCycleCount = 0;
//                randomIndices.Shuffle();
//            }
//        }

//        /// <summary>
//        /// <para>
//        /// Train the neural network using all the exemples
//        /// previously added with the method AddTrainingData.
//        /// </para>
//        /// <para>
//        /// For best results, call this method, 
//        /// then set the new inputs and update the neural network.
//        /// </para>
//        /// 
//        /// </summary>
//        /// <param name="learningRate">Learning rate.</param>
//        public void BatchTraining(float learningRate)
//        {
//            BatchTraining(learningRate, 0);
//        }

//        /// <summary>
//        /// <para>
//        /// Train the neural network using all the exemples
//        /// previously added with the method AddTrainingData.
//        /// </para>
//        /// <para>
//        /// For best results, call this method, 
//        /// then set the new inputs and update the neural network.
//        /// </para>
//        /// 
//        /// </summary>
//        public void BatchTraining()
//        {
//            BatchTraining(0.1f, 0);
//        }

//        #region MultiThread
//        Thread thread;
//        /// <summary>
//        /// Create a thread that will call BatchTraining in a loop. 
//        /// This method is optimal since it will learn in another thread
//        /// Making Unity run at full capacity.
//        /// 
//        /// </summary>
//        /// <param name="learningRate">Learning rate.</param>
//        /// <param name="momentum">Momentum.</param>
//        public void ThreadBatchTraining(float learningRate, float momentum)
//        {
//            if (!IsThreadAlive())
//                StartBatchThread(learningRate, momentum);
//        }

//        void StartBatchThread(float learningRate, float momentum)
//        {
//            thread = new Thread(() => LoopBatchTraining(learningRate, momentum));
//            thread.Start();
//        }

//        void LoopBatchTraining(float learningRate, float momentum)
//        {
//            while (IsThreadAlive())
//                BatchTraining(learningRate, momentum);
//        }

//        bool IsThreadAlive()
//        {
//            return (thread != null && thread.ThreadState != ThreadState.Unstarted && thread.ThreadState != ThreadState.Aborted);
//        }
//        /// <summary>
//        /// Closes the thread. Call this in void OnApplicationQuit().
//        /// Otherwise Unity will freeze.  
//        /// </summary>
//        public void CloseThread()
//        {
//            if (thread != null)
//            {
//                thread.Interrupt();
//                thread.Abort();
//                thread = null;
//                Debug.Log("thread closed");
//            }
//        }
//        #endregion

//        public class BatchInfo
//        {
//            public int sampleCount;
//        }
//    }
//}