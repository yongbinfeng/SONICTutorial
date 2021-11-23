---
description: document on how to prepare the model files
---

# Model Configuration

Triton's document on the model configuration can be found [here](https://github.com/triton-inference-server/server/blob/main/docs/model\_configuration.md), which includes a lot of details.&#x20;

The model directory should follow the structure like this:

```
deeptau_nosplit/
    config.pbtxt
    1/
        model.graphdef
```

where is the `deeptau_nosplit` model name, `config.pbtxt` is the configuration file, and `model.graphdef` under subdirectory `1` is the saved model file.&#x20;

The config file `config.pbtxt` follows a structure similar to the following:

```
name: "deeptau_nosplit"
platform: "tensorflow_graphdef"
max_batch_size: 1000
input [
  {
    name: "input_tau"
    data_type: TYPE_FP32
    dims: [47]
  },
  {
    name: "input_inner_egamma"
    data_type: TYPE_FP32
    dims: [ 11, 11, 86 ]
  },
  {
    name: "input_inner_muon"
    data_type: TYPE_FP32
    dims: [ 11, 11, 64 ]
  },
  {
    name: "input_inner_hadrons"
    data_type: TYPE_FP32
    dims: [ 11, 11, 38 ]
  },
  {
    name: "input_outer_egamma"
    data_type: TYPE_FP32
    dims: [ 21, 21, 86 ]
  },
  {
    name: "input_outer_muon"
    data_type: TYPE_FP32
    dims: [ 21, 21, 64 ]
  },
  {
    name: "input_outer_hadrons"
    data_type: TYPE_FP32
    dims: [ 21, 21, 38 ]
  }
]
output [
  {
    name: "main_output/Softmax"
    data_type: TYPE_FP32
    dims: [ 4 ]
  }
]
```

where `name` is the model name, `platform` is the model type. The names of the inputs and outputs tensors, their types, and also the shapes are specified in the `input` and `output`.

For models that support batching, the batch dimension is omitted in the model configuration. The shapes specified in the configuration files should start with the dimension after the batching.

For models with variable-length inputs, the shapes should be configured with -1. An example is like this:

```
name: "particlenet_AK4_PT"
platform: "pytorch_libtorch"
max_batch_size : 500
dynamic_batching {
   preferred_batch_size: [ 200 ]
}
input [
  {
    name: "pf_points__0"
    data_type: TYPE_FP32
    dims: [ 2, -1 ]
  },
  {
    name: "pf_features__1"
    data_type: TYPE_FP32
    dims: [ 20, -1 ]
  },
  {
    name: "pf_mask__2"
    data_type: TYPE_FP32
    dims: [ 1, -1 ]
  },
  {
    name: "sv_points__3"
    data_type: TYPE_FP32
    dims: [ 2, -1 ]
  },
  {
    name: "sv_features__4"
    data_type: TYPE_FP32
    dims: [ 11, -1 ]
  },
  {
    name: "sv_mask__5"
    data_type: TYPE_FP32
    dims: [ 1, -1 ]
  }
]
output [
  {
    name: "softmax__0"
    data_type: TYPE_FP32
    dims: [ 8 ]
    label_filename: "particlenet_labels.txt"
  }
]
```

**Note** for Pytorch models, the input and output names are arbitrary. What is crucial are the numbers at the end. A Pytorch model example can be found [here](https://github.com/fastmachinelearning/sonic-models/blob/master/models/particlenet\_AK4\_PT/config.pbtxt#L7-L45).

After preparing the model directory, you could test if it works by launching the Triton server with docker or singularity:

```
docker pull nvcr.io/nvidia/tritonserver:21.10-py3
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/full/path/to/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:21.10-py3 tritonserver --model-repository=/models
```

Some logs should be printed out and finally end with

```
+----------------------+---------+--------+
| Model                | Version | Status |
+----------------------+---------+--------+
| <model_name>         | <v>     | READY  |
| ..                   | .       | ..     |
| ..                   | .       | ..     |
+----------------------+---------+--------+
...
...
...
I1002 21:58:57.891440 62 grpc_server.cc:3914] Started GRPCInferenceService at 0.0.0.0:8001
I1002 21:58:57.893177 62 http_server.cc:2717] Started HTTPService at 0.0.0.0:8000
I1002 21:58:57.935518 62 http_server.cc:2736] Started Metrics Service at 0.0.0.0:8002
```
