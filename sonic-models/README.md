---
description: documents on the format of the model configuration and the optimizations
---

# SONIC Models

Currently SONIC uses [Nvidia Triton Inference Server](https://github.com/triton-inference-server/server) on the server side. It has a lot of nice features as explained in their git repo. Among these two of the most attractive features for us:

* it supports various backends: Tensorflow, TensorRT, Pytorch,ONNX, ScikitLearn, and also custom backends for non-ML algorithms
* Dynamic batching: dynamically batch the inference requests on the server side to make more efficient usage of the coprocessors

To deploy the models for Triton, we need to set up the model configurations. Examples of our current SONIC models can be found [here](https://github.com/fastmachinelearning/sonic-models/tree/master/models).

