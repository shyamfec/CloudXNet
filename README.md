# CloudXNet
CloudX-Net is an efficient and robust architecture used for detection of clouds from satellite images. It uses ASPP and Seperable Convolution to achieve high accuracy and efficiency. For details refer this paper. Here we share the CloudX-Net architecture (cxn_model.py). One of the dataset that we used to train the model can be downloaded from here.

The description of each file is as follows:

cxn_model.py - It contains the architecture and all the relevant functions required to run the architecture.
train & test.py - This code starts with reading the dataset provided above. Then it calls and complies the architecture from cxn_model.py followed by training and prediction.
metrics.py - It contains the codes for all the metrices given in the paper. The required inputs are Y_test(The groundtruth) and preds_test_t(The predictions).
For best implementation copy the different files to different cells of a google colab and run them in the order given above.

The code for the architecture is a modification of the CloudNet architecture by Sorour Mohajerani which can be found here.

