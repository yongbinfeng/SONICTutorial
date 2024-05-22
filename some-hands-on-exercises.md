---
description: hands-on exercises for computational HEP Traineeship Summer School
---

# Some hands-on exercises

For CMS students, log into the [LPC GPU nodes](./):

<pre class="language-bash"><code class="lang-bash">kinit username@FNAL.GOV
<strong>ssh username@cmslpcgpu[1-2].fnal.gov
</strong>export WorkDir=/uscms_data/d3/$USER/SONICHandsOn
export ContainerDir=/uscms_data/d3/yfeng/SONICTutorial/containers
</code></pre>

Local computers should also work fine if you have docker/podman/apptainer supports.

### Prepare files

Create the `WorkDir` and clone the relevant model and client code:

```bash
mkdir $WorkDir
cd $WorkDir
git clone git@github.com:yongbinfeng/TritonDemo.git
```

Both the server and clients images have been cloned under `$ContainerDir` on LPC node.

### Launch a server

```bash
BasePort=9000
apptainer run --nv -B $PWD/TritonDemo/models:/models $ContainerDir/triton_22.07.sif tritonserver --model-repository=/models --http-port=$BasePort --grpc-port=$((BasePort+1)) --metrics-port=$((BasePort+2))
```

The output should be similar to&#x20;

<details>

<summary>Server output</summary>

```log
I0522 19:43:55.032513 22 pinned_memory_manager.cc:240] Pinned memory pool is created at '0x7f0b8e000000' with size 268435456
I0522 19:43:55.035011 22 cuda_memory_manager.cc:105] CUDA memory pool is created on device 0 with size 67108864
I0522 19:43:55.051414 22 model_repository_manager.cc:1206] loading: add_sub:1
I0522 19:43:55.051571 22 model_repository_manager.cc:1206] loading: ExaTrk:1
I0522 19:43:55.051753 22 model_repository_manager.cc:1206] loading: particlenet_PT:1
I0522 19:43:56.800773 22 onnxruntime.cc:2458] TRITONBACKEND_Initialize: onnxruntime
I0522 19:43:56.800819 22 onnxruntime.cc:2468] Triton TRITONBACKEND API version: 1.10
I0522 19:43:56.801543 22 onnxruntime.cc:2474] 'onnxruntime' TRITONBACKEND API version: 1.10
I0522 19:43:56.801552 22 onnxruntime.cc:2504] backend configuration:
{"cmdline":{"auto-complete-config":"true","min-compute-capability":"6.000000","backend-directory":"/opt/tritonserver/backends","default-max-batch-size":"4"}}
I0522 19:43:56.840699 22 onnxruntime.cc:2560] TRITONBACKEND_ModelInitialize: ExaTrk (version 1)
I0522 19:43:56.842094 22 onnxruntime.cc:666] skipping model configuration auto-complete for 'ExaTrk': inputs and outputs already specified
I0522 19:43:58.359443 22 libtorch.cc:1917] TRITONBACKEND_Initialize: pytorch
I0522 19:43:58.359471 22 libtorch.cc:1927] Triton TRITONBACKEND API version: 1.10
I0522 19:43:58.360106 22 libtorch.cc:1933] 'pytorch' TRITONBACKEND API version: 1.10
I0522 19:43:58.360141 22 onnxruntime.cc:2603] TRITONBACKEND_ModelInstanceInitialize: ExaTrk (GPU device 0)
I0522 19:44:04.931652 22 python_be.cc:1767] TRITONBACKEND_ModelInstanceInitialize: add_sub_0 (CPU device 0)
I0522 19:44:04.932898 22 model_repository_manager.cc:1352] successfully loaded 'ExaTrk' version 1
I0522 19:44:05.232390 22 libtorch.cc:1966] TRITONBACKEND_ModelInitialize: particlenet_PT (version 1)
I0522 19:44:05.232494 22 model_repository_manager.cc:1352] successfully loaded 'add_sub' version 1
W0522 19:44:05.235038 22 libtorch.cc:262] skipping model configuration auto-complete for 'particlenet_PT': not supported for pytorch backend
I0522 19:44:05.235974 22 libtorch.cc:291] Optimized execution is enabled for model instance 'particlenet_PT'
I0522 19:44:05.235988 22 libtorch.cc:310] Cache Cleaning is disabled for model instance 'particlenet_PT'
I0522 19:44:05.235997 22 libtorch.cc:327] Inference Mode is disabled for model instance 'particlenet_PT'
I0522 19:44:05.236006 22 libtorch.cc:422] NvFuser is not specified for model instance 'particlenet_PT'
I0522 19:44:05.236684 22 libtorch.cc:2010] TRITONBACKEND_ModelInstanceInitialize: particlenet_PT (GPU device 0)
I0522 19:44:05.397227 22 model_repository_manager.cc:1352] successfully loaded 'particlenet_PT' version 1
I0522 19:44:05.397394 22 server.cc:559]
+------------------+------+
| Repository Agent | Path |
+------------------+------+
+------------------+------+

I0522 19:44:05.397478 22 server.cc:586]
+-------------+--------------------------------------------------------------+--------------------------------------------------------------+
| Backend     | Path                                                         | Config                                                       |
+-------------+--------------------------------------------------------------+--------------------------------------------------------------+
| python      | /opt/tritonserver/backends/python/libtriton_python.so        | {"cmdline":{"auto-complete-config":"true","min-compute-capab |
|             |                                                              | ility":"6.000000","backend-directory":"/opt/tritonserver/bac |
|             |                                                              | kends","default-max-batch-size":"4"}}                        |
| onnxruntime | /opt/tritonserver/backends/onnxruntime/libtriton_onnxruntime | {"cmdline":{"auto-complete-config":"true","min-compute-capab |
|             | .so                                                          | ility":"6.000000","backend-directory":"/opt/tritonserver/bac |
|             |                                                              | kends","default-max-batch-size":"4"}}                        |
|             |                                                              |                                                              |
| pytorch     | /opt/tritonserver/backends/pytorch/libtriton_pytorch.so      | {"cmdline":{"auto-complete-config":"true","min-compute-capab |
|             |                                                              | ility":"6.000000","backend-directory":"/opt/tritonserver/bac |
|             |                                                              | kends","default-max-batch-size":"4"}}                        |
+-------------+--------------------------------------------------------------+--------------------------------------------------------------+

I0522 19:44:05.399674 22 server.cc:629]
+----------------+---------+--------+
| Model          | Version | Status |
+----------------+---------+--------+
| ExaTrk         | 1       | READY  |
| add_sub        | 1       | READY  |
| particlenet_PT | 1       | READY  |
+----------------+---------+--------+

I0522 19:44:05.443770 22 metrics.cc:650] Collecting metrics for GPU 0: Tesla P100-PCIE-12GB
I0522 19:44:05.444129 22 tritonserver.cc:2176]
+----------------------------------+----------------------------------------------------------------------------------------------------------+
| Option                           | Value                                                                                                    |
+----------------------------------+----------------------------------------------------------------------------------------------------------+
| server_id                        | triton                                                                                                   |
| server_version                   | 2.24.0                                                                                                   |
| server_extensions                | classification sequence model_repository model_repository(unload_dependents) schedule_policy model_confi |
|                                  | guration system_shared_memory cuda_shared_memory binary_tensor_data statistics trace                     |
| model_repository_path[0]         | /models                                                                                                  |
| model_control_mode               | MODE_NONE                                                                                                |
| strict_model_config              | 0                                                                                                        |
| rate_limit                       | OFF                                                                                                      |
| pinned_memory_pool_byte_size     | 268435456                                                                                                |
| cuda_memory_pool_byte_size{0}    | 67108864                                                                                                 |
| response_cache_byte_size         | 0                                                                                                        |
| min_supported_compute_capability | 6.0                                                                                                      |
| strict_readiness                 | 1                                                                                                        |
| exit_timeout                     | 30                                                                                                       |
+----------------------------------+----------------------------------------------------------------------------------------------------------+

I0522 19:44:05.460750 22 grpc_server.cc:4608] Started GRPCInferenceService at 0.0.0.0:9001
I0522 19:44:05.463953 22 http_server.cc:3312] Started HTTPService at 0.0.0.0:9000
I0522 19:44:05.508576 22 http_server.cc:178] Started Metrics Service at 0.0.0.0:9002
```

</details>

where the three models of `particlenet` for jet flavor tagging (PyTorch model), `ExaTrk` for track reconstruction (ONNX model), and a toy `add_sub` model (Python code) have been loaded into the server.&#x20;

### Check model config files and server side metrics:

Try to take a look at the directory `$PWD/TritonDemo/models`, understand the structure, and browser some files (config.pbtxt and .py).

Now the server is basically ready for inference (with http port at `0.0.0.0:9000` and gPRC port at `0.0.0.0:9001`).&#x20;

Open another terminal, ssh to the GPU cluster, and run&#x20;

```bash
curl localhost:9002/metrics
```

What would you get?

<details>

<summary>Server metrics</summary>

```log
# HELP nv_inference_request_success Number of successful inference requests, all batch sizes
# TYPE nv_inference_request_success counter
nv_inference_request_success{model="particlenet_PT",version="1"} 0.000000
nv_inference_request_success{model="ExaTrk",version="1"} 0.000000
nv_inference_request_success{model="add_sub",version="1"} 0.000000
# HELP nv_inference_request_failure Number of failed inference requests, all batch sizes
# TYPE nv_inference_request_failure counter
nv_inference_request_failure{model="particlenet_PT",version="1"} 0.000000
nv_inference_request_failure{model="ExaTrk",version="1"} 0.000000
nv_inference_request_failure{model="add_sub",version="1"} 0.000000
# HELP nv_inference_count Number of inferences performed (does not include cached requests)
# TYPE nv_inference_count counter
nv_inference_count{model="particlenet_PT",version="1"} 0.000000
nv_inference_count{model="ExaTrk",version="1"} 0.000000
nv_inference_count{model="add_sub",version="1"} 0.000000
# HELP nv_inference_exec_count Number of model executions performed (does not include cached requests)
# TYPE nv_inference_exec_count counter
nv_inference_exec_count{model="particlenet_PT",version="1"} 0.000000
nv_inference_exec_count{model="ExaTrk",version="1"} 0.000000
nv_inference_exec_count{model="add_sub",version="1"} 0.000000
# HELP nv_inference_request_duration_us Cumulative inference request duration in microseconds (includes cached requests)
# TYPE nv_inference_request_duration_us counter
nv_inference_request_duration_us{model="particlenet_PT",version="1"} 0.000000
nv_inference_request_duration_us{model="ExaTrk",version="1"} 0.000000
nv_inference_request_duration_us{model="add_sub",version="1"} 0.000000
# HELP nv_inference_queue_duration_us Cumulative inference queuing duration in microseconds (includes cached requests)
# TYPE nv_inference_queue_duration_us counter
nv_inference_queue_duration_us{model="particlenet_PT",version="1"} 0.000000
nv_inference_queue_duration_us{model="ExaTrk",version="1"} 0.000000
nv_inference_queue_duration_us{model="add_sub",version="1"} 0.000000
# HELP nv_inference_compute_input_duration_us Cumulative compute input duration in microseconds (does not include cached requests)
# TYPE nv_inference_compute_input_duration_us counter
nv_inference_compute_input_duration_us{model="particlenet_PT",version="1"} 0.000000
nv_inference_compute_input_duration_us{model="ExaTrk",version="1"} 0.000000
nv_inference_compute_input_duration_us{model="add_sub",version="1"} 0.000000
# HELP nv_inference_compute_infer_duration_us Cumulative compute inference duration in microseconds (does not include cached requests)
# TYPE nv_inference_compute_infer_duration_us counter
nv_inference_compute_infer_duration_us{model="particlenet_PT",version="1"} 0.000000
nv_inference_compute_infer_duration_us{model="ExaTrk",version="1"} 0.000000
nv_inference_compute_infer_duration_us{model="add_sub",version="1"} 0.000000
# HELP nv_inference_compute_output_duration_us Cumulative inference compute output duration in microseconds (does not include cached requests)
# TYPE nv_inference_compute_output_duration_us counter
nv_inference_compute_output_duration_us{model="particlenet_PT",version="1"} 0.000000
nv_inference_compute_output_duration_us{model="ExaTrk",version="1"} 0.000000
nv_inference_compute_output_duration_us{model="add_sub",version="1"} 0.000000
# HELP nv_cache_num_entries Number of responses stored in response cache
# TYPE nv_cache_num_entries gauge
# HELP nv_cache_num_lookups Number of cache lookups in response cache
# TYPE nv_cache_num_lookups gauge
# HELP nv_cache_num_hits Number of cache hits in response cache
# TYPE nv_cache_num_hits gauge
# HELP nv_cache_num_misses Number of cache misses in response cache
# TYPE nv_cache_num_misses gauge
# HELP nv_cache_num_evictions Number of cache evictions in response cache
# TYPE nv_cache_num_evictions gauge
# HELP nv_cache_lookup_duration Total cache lookup duration (hit and miss), in microseconds
# TYPE nv_cache_lookup_duration gauge
# HELP nv_cache_insertion_duration Total cache insertion duration, in microseconds
# TYPE nv_cache_insertion_duration gauge
# HELP nv_cache_util Cache utilization [0.0 - 1.0]
# TYPE nv_cache_util gauge
# HELP nv_cache_num_hits_per_model Number of cache hits per model
# TYPE nv_cache_num_hits_per_model counter
nv_cache_num_hits_per_model{model="particlenet_PT",version="1"} 0.000000
nv_cache_num_hits_per_model{model="ExaTrk",version="1"} 0.000000
nv_cache_num_hits_per_model{model="add_sub",version="1"} 0.000000
# HELP nv_cache_hit_lookup_duration_per_model Total cache hit lookup duration per model, in microseconds
# TYPE nv_cache_hit_lookup_duration_per_model counter
nv_cache_hit_lookup_duration_per_model{model="particlenet_PT",version="1"} 0.000000
nv_cache_hit_lookup_duration_per_model{model="ExaTrk",version="1"} 0.000000
nv_cache_hit_lookup_duration_per_model{model="add_sub",version="1"} 0.000000
# HELP nv_cache_num_misses_per_model Number of cache misses per model
# TYPE nv_cache_num_misses_per_model counter
nv_cache_num_misses_per_model{model="particlenet_PT",version="1"} 0.000000
nv_cache_num_misses_per_model{model="ExaTrk",version="1"} 0.000000
nv_cache_num_misses_per_model{model="add_sub",version="1"} 0.000000
# HELP nv_cache_miss_lookup_duration_per_model Total cache miss lookup duration per model, in microseconds
# TYPE nv_cache_miss_lookup_duration_per_model counter
nv_cache_miss_lookup_duration_per_model{model="particlenet_PT",version="1"} 0.000000
nv_cache_miss_lookup_duration_per_model{model="ExaTrk",version="1"} 0.000000
nv_cache_miss_lookup_duration_per_model{model="add_sub",version="1"} 0.000000
# HELP nv_cache_miss_insertion_duration_per_model Total cache miss insertion duration per model, in microseconds
# TYPE nv_cache_miss_insertion_duration_per_model counter
nv_cache_miss_insertion_duration_per_model{model="particlenet_PT",version="1"} 0.000000
nv_cache_miss_insertion_duration_per_model{model="ExaTrk",version="1"} 0.000000
nv_cache_miss_insertion_duration_per_model{model="add_sub",version="1"} 0.000000
# HELP nv_gpu_utilization GPU utilization rate [0.0 - 1.0)
# TYPE nv_gpu_utilization gauge
nv_gpu_utilization{gpu_uuid="GPU-a07dfe08-b87f-1f96-7083-07a26d0fa83e"} 0.000000
# HELP nv_gpu_memory_total_bytes GPU total memory, in bytes
# TYPE nv_gpu_memory_total_bytes gauge
nv_gpu_memory_total_bytes{gpu_uuid="GPU-a07dfe08-b87f-1f96-7083-07a26d0fa83e"} 12884901888.000000
# HELP nv_gpu_memory_used_bytes GPU used memory, in bytes
# TYPE nv_gpu_memory_used_bytes gauge
nv_gpu_memory_used_bytes{gpu_uuid="GPU-a07dfe08-b87f-1f96-7083-07a26d0fa83e"} 554696704.000000
# HELP nv_gpu_power_usage GPU power usage in watts
# TYPE nv_gpu_power_usage gauge
nv_gpu_power_usage{gpu_uuid="GPU-a07dfe08-b87f-1f96-7083-07a26d0fa83e"} 33.246000
# HELP nv_gpu_power_limit GPU power management limit in watts
# TYPE nv_gpu_power_limit gauge
nv_gpu_power_limit{gpu_uuid="GPU-a07dfe08-b87f-1f96-7083-07a26d0fa83e"} 250.000000
# HELP nv_energy_consumption GPU energy consumption in joules since the Triton Server started
# TYPE nv_energy_consumption counter
nv_energy_consumption{gpu_uuid="GPU-a07dfe08-b87f-1f96-7083-07a26d0fa83e"} 0.000000
```

</details>

These metrics (computing time, cache size, GPU/CPU utilization etc) will be very useful to monitor. Can you identify these information?

### Run the actual inference

In the new GPU node, run the client container:

```bash
cd $WorkDir/TritonDemo
apptainer run $ContainerDir/triton_22.07.sdk.sif python3 test.py -m add_sub -u localhost:9001 -p grpc
```

What do you get? Check the code and see if you can run the inference using http and/or on particlenet model.

<details>

<summary>Test outputs:</summary>

```log
*******
 Result:
input0:  [ 0.6481743   1.4448979   0.2758574   0.4459793  -2.0501273   0.4608621
 -1.4055945  -1.2172754   0.6248972  -0.05864633 -0.12731536 -0.54622364
 -0.02361191  1.1548297  -0.91608065  0.00632136]
input1:  [-2.250999    0.1062035  -0.8693246  -0.9147433  -0.76667786 -0.33449128
  0.85628927  0.82451487 -0.67056346  0.59528124 -0.1628165   0.49658445
  1.5816904  -1.5217074   1.2673943   0.50004226]
sum:  [-1.6028247   1.5511014  -0.59346724 -0.468764   -2.8168051   0.12637082
 -0.5493052  -0.39276052 -0.04566628  0.5366349  -0.29013187 -0.0496392
  1.5580785  -0.36687768  0.35131365  0.50636363]
sub:  [ 2.8991733   1.3386943   1.145182    1.3607225  -1.2834494   0.7953534
 -2.2618837  -2.0417902   1.2954607  -0.65392756  0.03550114 -1.042808
 -1.6053023   2.676537   -2.183475   -0.4937209 ]
Passed all tests!
```

</details>

After running, you can pull the server side metrics again and see if there is any change.&#x20;

<details>

<summary>Server metrics:</summary>

```log
# HELP nv_inference_request_success Number of successful inference requests, all batch sizes
# TYPE nv_inference_request_success counter
nv_inference_request_success{model="particlenet_PT",version="1"} 10.000000
nv_inference_request_success{model="ExaTrk",version="1"} 0.000000
nv_inference_request_success{model="add_sub",version="1"} 15.000000
# HELP nv_inference_request_failure Number of failed inference requests, all batch sizes
# TYPE nv_inference_request_failure counter
nv_inference_request_failure{model="particlenet_PT",version="1"} 0.000000
nv_inference_request_failure{model="ExaTrk",version="1"} 0.000000
nv_inference_request_failure{model="add_sub",version="1"} 0.000000
# HELP nv_inference_count Number of inferences performed (does not include cached requests)
# TYPE nv_inference_count counter
nv_inference_count{model="particlenet_PT",version="1"} 10.000000
nv_inference_count{model="ExaTrk",version="1"} 0.000000
nv_inference_count{model="add_sub",version="1"} 15.000000
# HELP nv_inference_exec_count Number of model executions performed (does not include cached requests)
# TYPE nv_inference_exec_count counter
nv_inference_exec_count{model="particlenet_PT",version="1"} 10.000000
nv_inference_exec_count{model="ExaTrk",version="1"} 0.000000
nv_inference_exec_count{model="add_sub",version="1"} 15.000000
# HELP nv_inference_request_duration_us Cumulative inference request duration in microseconds (includes cached requests)
# TYPE nv_inference_request_duration_us counter
nv_inference_request_duration_us{model="particlenet_PT",version="1"} 7449014.000000
nv_inference_request_duration_us{model="ExaTrk",version="1"} 0.000000
nv_inference_request_duration_us{model="add_sub",version="1"} 50890.000000
# HELP nv_inference_queue_duration_us Cumulative inference queuing duration in microseconds (includes cached requests)
# TYPE nv_inference_queue_duration_us counter
nv_inference_queue_duration_us{model="particlenet_PT",version="1"} 454.000000
nv_inference_queue_duration_us{model="ExaTrk",version="1"} 0.000000
nv_inference_queue_duration_us{model="add_sub",version="1"} 253.000000
# HELP nv_inference_compute_input_duration_us Cumulative compute input duration in microseconds (does not include cached requests)
# TYPE nv_inference_compute_input_duration_us counter
nv_inference_compute_input_duration_us{model="particlenet_PT",version="1"} 776.000000
nv_inference_compute_input_duration_us{model="ExaTrk",version="1"} 0.000000
nv_inference_compute_input_duration_us{model="add_sub",version="1"} 279.000000
# HELP nv_inference_compute_infer_duration_us Cumulative compute inference duration in microseconds (does not include cached requests)
# TYPE nv_inference_compute_infer_duration_us counter
nv_inference_compute_infer_duration_us{model="particlenet_PT",version="1"} 7447242.000000
nv_inference_compute_infer_duration_us{model="ExaTrk",version="1"} 0.000000
nv_inference_compute_infer_duration_us{model="add_sub",version="1"} 49477.000000
# HELP nv_inference_compute_output_duration_us Cumulative inference compute output duration in microseconds (does not include cached requests)
# TYPE nv_inference_compute_output_duration_us counter
nv_inference_compute_output_duration_us{model="particlenet_PT",version="1"} 75.000000
nv_inference_compute_output_duration_us{model="ExaTrk",version="1"} 0.000000
nv_inference_compute_output_duration_us{model="add_sub",version="1"} 665.000000
# HELP nv_cache_num_entries Number of responses stored in response cache
# TYPE nv_cache_num_entries gauge
# HELP nv_cache_num_lookups Number of cache lookups in response cache
# TYPE nv_cache_num_lookups gauge
# HELP nv_cache_num_hits Number of cache hits in response cache
# TYPE nv_cache_num_hits gauge
# HELP nv_cache_num_misses Number of cache misses in response cache
# TYPE nv_cache_num_misses gauge
# HELP nv_cache_num_evictions Number of cache evictions in response cache
# TYPE nv_cache_num_evictions gauge
# HELP nv_cache_lookup_duration Total cache lookup duration (hit and miss), in microseconds
# TYPE nv_cache_lookup_duration gauge
# HELP nv_cache_insertion_duration Total cache insertion duration, in microseconds
# TYPE nv_cache_insertion_duration gauge
# HELP nv_cache_util Cache utilization [0.0 - 1.0]
# TYPE nv_cache_util gauge
# HELP nv_cache_num_hits_per_model Number of cache hits per model
# TYPE nv_cache_num_hits_per_model counter
nv_cache_num_hits_per_model{model="particlenet_PT",version="1"} 0.000000
nv_cache_num_hits_per_model{model="ExaTrk",version="1"} 0.000000
nv_cache_num_hits_per_model{model="add_sub",version="1"} 0.000000
# HELP nv_cache_hit_lookup_duration_per_model Total cache hit lookup duration per model, in microseconds
# TYPE nv_cache_hit_lookup_duration_per_model counter
nv_cache_hit_lookup_duration_per_model{model="particlenet_PT",version="1"} 0.000000
nv_cache_hit_lookup_duration_per_model{model="ExaTrk",version="1"} 0.000000
nv_cache_hit_lookup_duration_per_model{model="add_sub",version="1"} 0.000000
# HELP nv_cache_num_misses_per_model Number of cache misses per model
# TYPE nv_cache_num_misses_per_model counter
nv_cache_num_misses_per_model{model="particlenet_PT",version="1"} 0.000000
nv_cache_num_misses_per_model{model="ExaTrk",version="1"} 0.000000
nv_cache_num_misses_per_model{model="add_sub",version="1"} 0.000000
# HELP nv_cache_miss_lookup_duration_per_model Total cache miss lookup duration per model, in microseconds
# TYPE nv_cache_miss_lookup_duration_per_model counter
nv_cache_miss_lookup_duration_per_model{model="particlenet_PT",version="1"} 0.000000
nv_cache_miss_lookup_duration_per_model{model="ExaTrk",version="1"} 0.000000
nv_cache_miss_lookup_duration_per_model{model="add_sub",version="1"} 0.000000
# HELP nv_cache_miss_insertion_duration_per_model Total cache miss insertion duration per model, in microseconds
# TYPE nv_cache_miss_insertion_duration_per_model counter
nv_cache_miss_insertion_duration_per_model{model="particlenet_PT",version="1"} 0.000000
nv_cache_miss_insertion_duration_per_model{model="ExaTrk",version="1"} 0.000000
nv_cache_miss_insertion_duration_per_model{model="add_sub",version="1"} 0.000000
# HELP nv_gpu_utilization GPU utilization rate [0.0 - 1.0)
# TYPE nv_gpu_utilization gauge
nv_gpu_utilization{gpu_uuid="GPU-a07dfe08-b87f-1f96-7083-07a26d0fa83e"} 0.000000
# HELP nv_gpu_memory_total_bytes GPU total memory, in bytes
# TYPE nv_gpu_memory_total_bytes gauge
nv_gpu_memory_total_bytes{gpu_uuid="GPU-a07dfe08-b87f-1f96-7083-07a26d0fa83e"} 12884901888.000000
# HELP nv_gpu_memory_used_bytes GPU used memory, in bytes
# TYPE nv_gpu_memory_used_bytes gauge
nv_gpu_memory_used_bytes{gpu_uuid="GPU-a07dfe08-b87f-1f96-7083-07a26d0fa83e"} 699400192.000000
# HELP nv_gpu_power_usage GPU power usage in watts
# TYPE nv_gpu_power_usage gauge
nv_gpu_power_usage{gpu_uuid="GPU-a07dfe08-b87f-1f96-7083-07a26d0fa83e"} 32.768000
# HELP nv_gpu_power_limit GPU power management limit in watts
# TYPE nv_gpu_power_limit gauge
nv_gpu_power_limit{gpu_uuid="GPU-a07dfe08-b87f-1f96-7083-07a26d0fa83e"} 250.000000
# HELP nv_energy_consumption GPU energy consumption in joules since the Triton Server started
# TYPE nv_energy_consumption counter
nv_energy_consumption{gpu_uuid="GPU-a07dfe08-b87f-1f96-7083-07a26d0fa83e"} 0.000000
```

</details>

### Running Perf Client

Keep server running. Launch the client and run perf client

```bash
apptainer run $ContainerDir/triton_22.07.sdk.sif perf_analyzer \
    -m particlenet_PT --percentile=95 -i grpc --async -p 9000 \
    --shape pf_points__0:2,50 --shape pf_features__1:25,50 --shape pf_mask__2:1,50 --shape sv_points__3:2,4 --shape sv_features__4:11,4 --shape sv_mask__5:1,4\
     -u localhost:9001 -i grpc --concurrency-range 1:1 -b 10
```

Do you understand this command? What is the output?

<details>

<summary>Perf Client output</summary>

```log
*** Measurement Settings ***
  Batch size: 10
  Using "time_windows" mode for stabilization
  Measurement window: 9000 msec
  Using asynchronous calls for inference
  Stabilizing using p95 latency

Request concurrency: 1
  Client:
    Request count: 8452
    Throughput: 2608.5 infer/sec
    p50 latency: 3787 usec
    p90 latency: 3900 usec
    p95 latency: 3938 usec
    p99 latency: 4078 usec
    Avg gRPC time: 3798 usec ((un)marshal request/response 7 usec + response wait 3791 usec)
  Server:
    Inference count: 84520
    Execution count: 8452
    Successful request count: 8452
    Avg request latency: 3584 usec (overhead 48 usec + queue 30 usec + compute input 66 usec + compute infer 2922 usec + compute output 517 usec)

Inferences/Second vs. Client p95 Batch Latency
Concurrency: 1, throughput: 2608.5 infer/sec, latency 3938 usec
```

</details>

Check what happends if you increase the batch size and concurrency. One can also log into the same GPU node with a new terminal and monitor the GPU usage with

```bash
watch -n 1 nvidia-smi
```

One example output is

<details>

<summary>nvidia-smi</summary>

```log
Wed May 22 14:58:03 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15	   CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla P100-PCIE-12GB           Off |   00000000:65:00.0 Off |                    0 |
| N/A   49C    P0             82W /  250W |     990MiB /  12288MiB |	 65%	  Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage	  |
|=========================================================================================|
|    0   N/A  N/A    408322	 C   tritonserver                                  988MiB |
+-----------------------------------------------------------------------------------------+
```

</details>

Do you understand the ouput?

### Optimizing models with perf client

Model inference can be optimized through dynamic batching sizes, numer of model instances, TensorRT optimizations, Just-in-time compilations, and reduced precision/quantization. Some of these are already supported and easily checked. More information are provided [here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user\_guide/optimization.html).

#### Model instances

Edit the particlenet model config file `$WorkDir/TritonDemo/models/particlenet_PT/config.pbtxt` and add following lines:

```
instance_group [ { count: 2 }]
```

to the end of the config file. Kill the server and relaunch it. Run the perf client on particlenet again. Compare the model performance. Do you see any difference. Also try increasing the batch size?

#### TensorRT optimizations

Edit the config file and add

```
optimization { execution_accelerators {
  gpu_execution_accelerator : [ {
    name : "tensorrt"
    parameters { key: "precision_mode" value: "FP16" }
    parameters { key: "max_workspace_size_bytes" value: "1073741824" }
    }]
}}
```

to enable reduced precision with TensorRT optimization. Run the perf client again. Do you see any difference?

More optimizations can be tested, depending on the ML model, backends, IO size, etc.
