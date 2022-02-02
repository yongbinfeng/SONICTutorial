---
description: >-
  Small demo for running inference as a service with Triton server and the
  python client, with no CMSSW involved yet.
---

# Quick Triton Demo

First download all the needed containers following the instructions in the [Prerequisites page](prerequisites.md).

Clone the small demo [repository](https://github.com/yongbinfeng/TritonDemo):

```
git clone git@github.com:yongbinfeng/TritonDemo.git
```

Launch the triton server with the model \`add\_sub\` saved in the repository:

```
docker run -it --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/Path_To_TritonDemo/models/:/models nvcr.io/nvidia/tritonserver:21.10-py3 tritonserver --model-repository=/models/
```

replace the \`Path\_To\_TritonDemo\` with your local path. Once you get:

```
I0202 06:41:02.923277 1 server.cc:538]
+-----------------+---------+--------+
| Model           | Version | Status |
+-----------------+---------+--------+
| add_sub         | 1       | READY  |
| particlenet_AK4 | 1       | READY  |
+-----------------+---------+--------+
...
I0127 20:32:59.322383 1 grpc_server.cc:4117] Started GRPCInferenceService at 0.0.0.0:8001
I0127 20:32:59.323244 1 http_server.cc:2815] Started HTTPService at 0.0.0.0:8000
I0127 20:32:59.365762 1 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```

it means the server is well setup and ready for inference requests.

Then launch a client container and run the test script:

```
docker run -it --gpus=1 --rm --net=host -v/Path_To_TritonDemo:/demo nvcr.io/nvidia/tritonserver:21.10-py3-sdk
cd /demo
python test.py
```

To run with the ParticleNet model,&#x20;

```
python test.py -m "particlenet_AK4"
```

**Note** the server has to be changed to some elder version to properly support ONNX models. Tested and it works well with `nvcr.io/nvidia/tritonserver:21.02-py3`
