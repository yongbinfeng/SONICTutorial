---
description: List of prerequisites to run SONIC + Triton
---

# Prerequisites

Usually we run Triton server (and clients) with **Docker** or **Singularity**, since Triton developers have provided the well-compiled version to be directly used.&#x20;

### Run with docker/podman

To run the server with docker or podman, first pull the docker image:

```
docker pull nvcr.io/nvidia/tritonserver:22.07-py3
```
or the equivalent of 
```
podman pull nvcr.io/nvidia/tritonserver:22.07-py3
```
for podman, where `22.07` is the Triton version you want to use. Other releases can be found [here](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html) and [here](https://github.com/triton-inference-server/server/releases). Better to check the backend (Tensorflow, Pytorch, ONNX, etc.) versions when choosing the right release.

Sometimes the Triton clients are also needed, e.g., to run [`perf_client` tests](https://github.com/triton-inference-server/server/blob/main/docs/perf\_analyzer.md) for model optimizations. To do this, the client docker image would also be needed:

```
docker pull nvcr.io/nvidia/tritonserver:22.07-py3-sdk
```

Similarly, `22.07` can be changed to other releases.

Then you can run the Triton server via

```
docker run -it --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/path/to/triton/models/:/models nvcr.io/nvidia/tritonserver:22.07-py3 tritonserver --model-repository=/models/
```

remove the `--gpus=1` option if you want to run on CPUs.

To run the Triton clients, do

```
docker run -it --gpus=1 --rm --net=host -v/path/to/triton/models:/models nvcr.io/nvidia/tritonserver:22.07-py3-sdk
```

where the `--net=host` argument is to use the host networking, so that you can ping the server with the correct ports running in a different container.

### Run with Singularity/Apptainer

&#x20;Some of the clusters do not support docker/podman but Singularity (or Apptainer). To run the images with Singularity, similar to the docker case, pull the image first:

```
apptainer pull triton_22.07.sif docker://nvcr.io/nvidia/tritonserver:22.07-py3
```

which will pull the image from docker and save it as a `sif` file. More information can be found [here](https://sylabs.io/guides/3.2/user-guide/cli/singularity\_pull.html). Sometimes one might run into disk quota issues, as the default cache directory is under HOME directory. In this case, try running with

```
APPTAINER_CACHEDIR=/directory/with/more/disk apptainer pull triton_22.07.sif docker://nvcr.io/nvidia/tritonserver:22.07-py3
```

as provided [here](https://docs.sylabs.io/guides/3.3/user-guide/build\_env.html). Similar commands for Apptainer.

Then run the container with:

```
apptainer run --nv -B /path/to/triton/repo:/models triton_22.07.sif tritonserver --model-repository=/models
```

remove the `--nv` option if not running on GPUs. The default ports for Triton are `8000`, `8001`, and `8002` for HTTP, gRPC, and metrics, respectively. To change these, use
```
BasePort=9000
apptainer run --nv -B /path/to/triton/repo:/models triton_22.07.sif tritonserver --model-repository=/models --http-port=$BasePort --grpc-port=$((BasePort+1)) --metrics-port=$((BasePort+2))
```

### Run with cmsTriton scripts

(To be updated.)
