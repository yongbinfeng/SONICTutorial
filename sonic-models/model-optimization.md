---
description: documents on how to optimize the model deployments
---

# Model optimization

Triton offers a lot of nice ways to better utilize the coprocessors (in our case probably GPUs), such as dynamic batching, concurrent model executions, TensorRT optimizations, etc.&#x20;

### Dynamic Batching&#x20;

The dynamic batching feature can be enabled by adding to the model config file

```
dynamic_batching {
    preferred_batch_size: [ 4, 8 ]
    max_queue_delay_microseconds: 100
}
```

An example can be found [here](https://github.com/fastmachinelearning/sonic-models/blob/master/models/deeptau\_nosplit/config.pbtxt#L48-L50). The preferred batch size and max delay time can be explored via perf client below. The official document with Triton is [here](https://github.com/triton-inference-server/server/blob/main/docs/model\_configuration.md#dynamic-batcher).

### Model Instances

Sometimes one model instance on one GPU is not able to saturates the GPU, and one could explore if increasing the number of the model instances on one GPU to increase the throughput. The model instances can be configured in the model config file:

```
instance_group [
    {
      count: 2
      kind: KIND_GPU
    }
]
```

which could creat two identical model instances can run in paralle on the same GPU. More documents on this can be found [here](https://github.com/triton-inference-server/server/blob/main/docs/model\_configuration.md#instance-groups).

### TensorRT Optimization

ML model inference throughputs can be accelerated by converting them into TensorRT models. Triton currently offers TensorRT optimizations for ONNX and Tensorflow models. To implement this simply add to the model config file:

```
optimization { execution_accelerators {
  gpu_execution_accelerator : [ {
    name : "tensorrt"
    parameters { key: "precision_mode" value: "FP32" }}]
}}
```

Note this would require some preprocessing and warmup time for the first inference request. More documents on this can be found [here](https://github.com/triton-inference-server/server/blob/main/docs/optimization.md#framework-specific-optimization).

### Run tests with Perf Client

One can explore the optimal model configuration for deployment by running throughput and latency tests and benchmarks via [perf client](https://github.com/triton-inference-server/server/blob/main/docs/user\_guide/perf\_analyzer.md) (also called ``perf analyzer'').&#x20;

To run perf client, first launch the Triton server serving the models via

```bash
docker run -it --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/path/to/triton/models/:/models nvcr.io/nvidia/tritonserver:21.10-py3 tritonserver --model-repository=/models/
```

then run the Triton client via

```bash
docker run -it --gpus=1 --rm --net=host -v/path/to/triton/models:/models nvcr.io/nvidia/tritonserver:21.10-py3-sdk
```

Inside the Triton client container, exectuate the command

```bash
perf_analyzer -m deeptau_nosplit --percentile=95 -u localhost:8001 -i grpc --async -p 9000 --concurrency-range 4:4 -b 100
```

which will generate some random inputs with batch size 100 and concurrency 4 for deeptau\_nosplit and ping the server at localhost:8001 visa grpc protocol. The outputs are similar to

```
Request concurrency: 4
  Client:
    Request count: 415
    Throughput: 83 infer/sec
    Avg latency: 48064 usec (standard deviation 6412 usec)
    p50 latency: 47975 usec
    p90 latency: 56670 usec
    p95 latency: 59118 usec
    p99 latency: 63609 usec
    Avg HTTP time: 48166 usec (send/recv 264 usec + response wait 47902 usec)
  Server:
    Inference count: 498
    Execution count: 498
    Successful request count: 498
    Avg request latency: 45602 usec (overhead 39 usec + queue 33577 usec + compute input 217 usec + compute infer 11753 usec + compute output 16 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 4, throughput: 83 infer/sec, latency 48064 usec
```

which includes the throughput, latency, and some metrics on the server side. The outputs can be saved into csv files by adding `-f output.csv` to the end.

One example script to loop over some models and varying the batch size is attached here:

```python
import os
from collections import OrderedDict
import subprocess

commands = OrderedDict()
commands['particlenet_AK4'] = "perf_analyzer -m particlenet_AK4 --percentile=95 -u localhost:8021 --shape pf_points:2,50 --shape pf_features:20,50 --shape pf_mask:1,50 --shape sv_points:2,4 --shape sv_features:11,4 --shape sv_mask:1,4 -i grpc --async -p 9000 --concurrency-range 4:4 -b"
commands['deeptau_nosplit'] = "perf_analyzer -m deeptau_nosplit --percentile=95 -u localhost:8021 -i grpc --async -p 9000 --concurrency-range 4:4 -b"

outdir = "/output/"

for model, command in commands.items():
    outdir = f"/output/{model}"
    if not os.path.isdir(outdir):
        subprocess.run(f"mkdir -p {outdir}", shell=True)
    for batch in [1, 4, 8, 16, 32, 64, 128]:
        tmp = command
        tmp += f" {batch} -f {outdir}/{model}_batch{batch}.csv"
        print(tmp)
        subprocess.run(f"{tmp}", shell=True)
```

This loops over two models and measures the throughputs and latency by varying the batch sizes, with the outputs saved to `{outdir}/{model}/{model}_{batch}.csv`. One can scan different concurrencies, model instances, etc and compare the performance using a similar setup. 

Then the outputs can be visualized using the csv files. One example script is provide here:

```python
import csv
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
# plotting style
import mplhep as hep
plt.style.use(hep.style.CMS)

throughputs = OrderedDict()
latencys = OrderedDict()

for model in ['particlenet_AK4', 'particlenet_AK4_PT']:
    dirname = f'/home/yfeng/results/{model}'
    if not os.path.isdir(dirname):
        continue
    throughputs[model] = OrderedDict()
    latencys[model] = OrderedDict()

    for batch in [1, 4, 8, 16, 32, 64, 128]:
        fname = f'{dirname}/{model}_batch{batch}.csv'
        if not os.path.isfile(fname):
            continue
        with open (fname, newline='') as csvfile:
            datareader = csv.reader(csvfile)
            for i, row in enumerate(datareader):
                if i!=1:
                    continue
                print(row)
                print(row[0])
            throughputs[model][batch] = float(row[1])
            latencys[model][batch] = float(row[-1])/1.0e3

labels = {
    'particlenet_AK4': 'PN-AK4 ONNX',
    'deeptau_nosplit': 'DeepTau',
}

markers = ['*', '^', 's', '1']

# make some throughputs and latency plots
def Plot(models, outputname):
    if len(models) == 0:
        return
    model = models[0]
    batchs = np.array(list(throughputs[model].keys()), dtype=np.float32)

    vals_throughput = [np.array(list(throughputs[model].values())) for model in models]
    vals_latency = [np.array(list(latencys[model].values())) for model in models]

    legends = [labels[model] for model in models]

    fig, axs = plt.subplots(2,2, figsize=(20,8), gridspec_kw={'height_ratios': [3, 1]})
    axs[0, 0].plot([], [], ' ', label="Tesla T4")
    axs[1, 0].plot([], [], ' ', label="Tesla T4")
    axs[0, 1].plot([], [], ' ', label="Tesla T4")
    axs[1, 1].plot([], [], ' ', label="Tesla T4")

    for i, model in enumerate(models):
        axs[0, 0].plot(batchs, vals_throughput[i], label=legends[i], marker=markers[i], markersize=10)
        axs[0, 1].plot(batchs, vals_latency[i],    label=legends[i], marker=markers[i], markersize=10)

    for i, model in enumerate(models):
        axs[1, 0].plot(batchs, vals_throughput[i]/vals_throughput[0])
        axs[1, 1].plot(batchs, vals_latency[i]/vals_latency[0])

    axs[1, 0].set_xlabel('Batch Size')
    axs[1, 0].set_ylabel('Ratio')
    axs[1, 0].set_xscale('log')
    axs[0, 0].set_ylabel('Throughput [evt/s]')
    axs[0, 0].set_xticklabels([])
    axs[0, 0].set_xscale('log')
    axs[0, 0].legend()

    axs[1, 1].set_xlabel('Batch Size')
    axs[1, 1].set_ylabel('Ratio')
    axs[1, 1].set_xscale('log')
    axs[0, 1].set_ylabel('Latency [ms]')
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_xticklabels([])
    axs[0, 1].set_xscale('log')
    axs[0, 1].legend()

    fig.subplots_adjust(hspace=0)
    fig.savefig(f"{outputname}.png")
Plot(['particlenet_AK4', 'deeptau_nosplit'], 'Test')
```
One example for DeepTau\_nosplit is shown in the following point:&#x20;

![Throughputs and latency vs batch size from perf client tests](<../.gitbook/assets/image (2).png>)

For models with variable-length inputs, the input dimensions need to be fully configured for running perf client with random or zero inputs. An example command for running ParticleNet model:
```bash
perf_analyzer -m particlenet_AK4 --percentile=95 -u localhost:8021 --shape pf_points:2,50 --shape pf_features:20,50 --shape pf_mask:1,50 --shape sv_points:2,4 --shape sv_features:11,4 --shape sv_mask:1,4 -i grpc --async -p 9000 --concurrency-range 4:4 -b 100
```
Passing a json file with inputs are also supported with `Perf Client`. More details can be found from the NVIDIA Triton User Guide.
