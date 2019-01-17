This code includes the detailed implementation of the paper:

Reference:
Shengke Xue, Xinyu Jin, "Robust Classwise and Projective Low-Rank Representation 
for Image Classification", Signal, Image and Video Processing, 2018, 12(1):107-115. 
DOI 10.1007/s11760-017-1136-1

It is partially composed of RASL code implementation,
Reference:
Peng, Y., et al., RASL, IEEE Trans. on PAMI, 2012.

You may start an example by directly running 'example_*.m' for various datasets.
|--------------
|-- example_COIL.m
|-- example_dummy.m       figure 2 in our paper
|-- example_GTF.m
|-- example_MNIST.m       figure 3, Table 1 in our paper
|-- example_ORL.m
|-- example_ORL.m         figure 6, 7 in our paper
|-- example_Yale.m        figure 4, 5 in our paper
|-------------

The code contains:
|--------------
|-- CPLR_func/                CPLR functions training projection
    |-- projection_predict.m  predict labels for test images
    |-- projection_train.m    train projection using GDM
|-- data/                     directory for image datasets *.mat
|-- figure/                   directory for saving figures
|-- load_func/                functions for loading datasets
|-- RASL_func/                RASL functions for training alignment
    |-- RASL_inner_ialm.m     inexact ALM algorithm of RASL
    |-- RASL_main.m           main loop of RASL
    |-- RASL_plot.m           display some image results after training
|-- result/                   directory for saving experimental results
|-- util/                     fundamental functions
|-------------

Note that, our code adopts the inexact augmented Lagrange multiplier (ALM) 
method and the gradient descent method with momentum (GDM) throughout this 
CPLRR implementation.

For algorithm details, please read our paper, in which we interpret more details.

If you have any questions about this implementation, please do not hesitate to contact us. 

Shengke Xue, 
College of Information Science and Electronic Engineering,
Zhejiang University, Hangzhou, P. R. China,
e-mail: (either one is o.k.)
xueshengke@zju.edu.cn, xueshengke1993@gmail.com.