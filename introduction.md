---
description: Quick introduction to SONIC
---

# Introduction

Readers are strongly encouraged to go through the [introductory slides](files/SONIC\_Introduction\_MLHATS.pdf) first!&#x20;

SONIC is the acronym for Service for Optimized Network Inference on Coprocessors. It is based on inference as a service. Instead of the usual case where the coprocessors (GPUs, FPGAs, ASICs) are directly connected to the CPUs, as-a-Service connects the CPUs and coprocessors **via networks**. As shown in the diagram below, with as-a-Service computing, clients only need to communicate with the server and handle the IOs, and the servers will direct the coprocessors for computing.

![Direct connection vs as-a-service](<.gitbook/assets/image (1) (1).png>)

As-a-service computing has a lot of benefits. For example:

* **The ML framework is factorized out of the client-side code**. To run the ML inferences, supports for Tensorflow, Pytorch, ONNX, XGBoost, etc are required. With as-a-service computing, servers will support these different frameworks and clients only need to handle the IOs.
* **Simple support for different coprocessors**. Client-side code does not need to be changed for different coprocessors on the server side.
* **Multiple clients can communicate with one coprocessor**. Often coprocessors are very powerful and one CPU client is not sufficient to saturate the coprocessor. With as-a-service we can make more efficient use of the coprocessors by increasing the number of clients.
* **Multiple coprocessors can communicate with one CPU client**. One CPU client might need to run multiple inferences on different models. These models can be deployed on different coprocessors to make better usage.&#x20;

In CMSSW, we set up the [SONIC workflow](https://github.com/cms-sw/cmssw/tree/master/HeterogeneousCore/SonicCore) to run inference as a service. The clients are deployed in CMSSW to handle the IOs; [Nvidia Triton Inference server](https://github.com/triton-inference-server/server) is chosen to run inferences for Machine-Learning models (and also classical domain algorithms), as shown below in the diagram.

![SONIC in CMSSW](<.gitbook/assets/image (1).png>)

Some useful links to the relevant papers, documents, and code, and also some step-by-step instructions to set up the model and producers are provided in the following sections.
