# Narrow-Beam-Prediction-using-Neural-Network-Decoder

**Introduction​**

This repo contains the source code for refined beam prediction with wide-beam-measurement which uses Neural Network Decoder algorithm to predict the indices of the optimal refined beams.

​

**Description (Simulation Environment and AI Algorithm)​**

**Dependencies:**

-   Python 3.8-3.11
-   Torch
-   Numpy

**Code Description:**

-   **DataProcessing.py:** This wrapper contains preprocessing requirements converting the matlab data into the .npz format
-   **DistanceDecoding.py**: This wrapper contains testing for the refined beam prediction with wide-beam measurement narrow beam prediction and the reference approach.
-   **Train.py** :

-   This wrapper contains preprocessing functions as follows:

1.  Generate dataSet
2.  Generate data loader
3.  Split the data for train, validate
4.  Call trainer.py for batch sample training
5.  Validate model performance
6.  Save model

-   **Trainer.py**:

-   This wrapper contains inheritance of torch.nn.Module in Tarin.py as follows:

1.  Define the optimizer
2.  Define the scheduler
3.  Initiate NN model
4.  Pass the data to the model for training
5.  Return the loss to train.py\
​

**Example Usage​**

We examine three wide beam codebook designs: the common codebook for wide beam codebook design (WB) , wide beam codebook design incorporating a circular-shift operation (CSWB) , and wide beam codebook design featuring partial random coding (PR-WB). For the decoding technique, we consider two methods: the proposed ML-based decoder and the non-ML method (DD). As illustrated in Fig.1, the cumulative distribution function (CDF) of eRSRP is computed using the testing data. Additionally, the y-axis value at eRSRP∼=0 dB indicates the prediction accuracy. The results indicate that: (1) the ML decoder significantly outperforms DD across all wide beam codebook designs; (2) the proposed CSWB and PR-WB models surpass the WB for each decoder method; (3) In our configuration, CS-WB with DD and PR-WB with DD demonstrate comparable performance to WB withML, highlighting the importance of the wide beam design for refined beam prediction.

![image](https://github.com/user-attachments/assets/29238747-9fff-4707-8a9a-006b8c24ad0f)


