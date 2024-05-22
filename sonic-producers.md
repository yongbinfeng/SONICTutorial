---
description: Example and instructions on preparing a SONIC producer
---

# SONIC Producers in CMS

In CMSSW, SONIC Triton producers handle the communications between the server and the client (i.e., CMSSW itself). As explained in the [Introduction](introduction.md) section, the producer sends the inference request together with the model inputs via `acquire` function, and receives the inference outputs from the server via `produce`. A more detailed explanation is provided [here](https://github.com/cms-sw/cmssw/tree/master/HeterogeneousCore/SonicTriton#modules).

### Producer Example: DeepMET

Now we take [`DeepMETSonicProducer`](https://github.com/fastmachinelearning/cmssw/blob/CMSSW\_12\_0\_0\_pre5\_SONIC/RecoMET/METPUSubtraction/plugins/DeepMETSonicProducer.cc#L46) as a real example and go through the implementation.&#x20;

#### Model config

The model config file can be found [here](https://github.com/fastmachinelearning/sonic-models/blob/master/models/deepmet/config.pbtxt). It requires four input tensors, with the tensor names, shapes, and data types shown below:

```
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 4500, 8 ]
  },
  {
    name: "input_cat0"
    data_type: TYPE_FP32
    dims: [ 4500, 1 ]
  },
  {
    name: "input_cat1"
    data_type: TYPE_FP32
    dims: [ 4500, 1 ]
  },
  {
    name: "input_cat2"
    data_type: TYPE_FP32
    dims: [ 4500, 1 ]
  }
]
```

The output of the model is a two-dimensional tensor, with the name `output/BiasAdd`:

```
output [
  {
    name: "output/BiasAdd"
    data_type: TYPE_FP32
    dims: [ 2 ]
  }
]
```

#### SONIC producer

In the SONIC producer, the **batch size** is configured [here](https://github.com/fastmachinelearning/cmssw/blob/CMSSW\_12\_0\_0\_pre5\_SONIC/RecoMET/METPUSubtraction/plugins/DeepMETSonicProducer.cc#L46):

```cpp
client_->setBatchSize(1);
```

The batch size is set to 1 since there is only one "MET" per event. For the inferences on each physics object such as jets, taus, etc. The batch size should be set to the number of these objects in one event, examples can be found [here](https://github.com/fastmachinelearning/cmssw/blob/CMSSW\_12\_0\_0\_pre5\_SONIC/RecoTauTag/RecoTau/plugins/DeepTauIdSonicProducer.cc#L258) and [here](https://github.com/fastmachinelearning/cmssw/blob/CMSSW\_12\_0\_0\_pre5\_SONIC/RecoBTag/ONNXRuntime/plugins/ParticleNetSonicJetTagsProducer.cc#L132).

The **inputs** are located [here](https://github.com/fastmachinelearning/cmssw/blob/CMSSW\_12\_0\_0\_pre5\_SONIC/RecoMET/METPUSubtraction/plugins/DeepMETSonicProducer.cc#L52-L66):

```cpp
  auto& input = iInput.at("input");
  auto pfdata = input.allocate<float>();
  auto& vpfdata = (*pfdata)[0];

  auto& input_cat0 = iInput.at("input_cat0");
  auto pfchg = input_cat0.allocate<float>();
  auto& vpfchg = (*pfchg)[0];

  auto& input_cat1 = iInput.at("input_cat1");
  auto pfpdgId = input_cat1.allocate<float>();
  auto& vpfpdgId = (*pfpdgId)[0];

  auto& input_cat2 = iInput.at("input_cat2");
  auto pffromPV = input_cat2.allocate<float>();
  auto& vpffromPV = (*pffromPV)[0];
```

`input.allocate<float>()` would return a pointer to `vector<vector<float>>`, with the [reserved size](https://github.com/cms-sw/cmssw/blob/master/HeterogeneousCore/SonicTriton/src/TritonData.cc#L147-L160) of `batch_size*sizeShape`, where the `sizeShape` is the product of shape dimension (in this case `sizeShape=4500*8` .&#x20;

For tensors with variable inputs, the shapes need to be set via `setShape`. One example is [here](https://github.com/fastmachinelearning/cmssw/blob/CMSSW\_12\_0\_0\_pre5\_SONIC/RecoBTag/ONNXRuntime/plugins/ParticleNetSonicJetTagsProducer.cc#L160).

The first dimension of `pfdata` is the batch, and the second dimension is the `sizeShape` for the tensor in one batch. In the DeepMET case there is only one batch per event, so we only need the first element:

```cpp
  auto& vpfdata = (*pfdata)[0];
  auto& vpfchg = (*pfchg)[0];
  auto& vpfpdgId = (*pfpdgId)[0];
  auto& vpffromPV = (*pffromPV)[0];
```

Then we loop over all the PF candidates and fill in the inputs:

```cpp
  for (const auto& pf : pfs) {
    ...

    // PF keys [b'PF_dxy', b'PF_dz', b'PF_eta', b'PF_mass', b'PF_pt', b'PF_puppiWeight', b'PF_px', b'PF_py']
    vpfdata.push_back(pf.dxy());
    vpfdata.push_back(pf.dz());
    vpfdata.push_back(pf.eta());
    vpfdata.push_back(pf.mass());
    vpfdata.push_back(scale_and_rm_outlier(pf.pt(), scale_));
    vpfdata.push_back(pf.puppiWeight());
    vpfdata.push_back(scale_and_rm_outlier(pf.px(), scale_));
    vpfdata.push_back(scale_and_rm_outlier(pf.py(), scale_));

    vpfchg.push_back(charge_embedding.at(pf.charge()));

    vpfpdgId.push_back(pdg_id_embedding.at(pf.pdgId()));

    vpffromPV.push_back(pf.fromPV());
    ...
  }
```

The input data are finally sent to the server:

```cpp
  input.toServer(pfdata);
  input_cat0.toServer(pfchg);
  input_cat1.toServer(pfpdgId);
  input_cat2.toServer(pffromPV);
```

These are the essential elements to prepare the `acquire` function in the SONIC producer.

CMSSW receives the inference **outputs** from the server by `produce`. For DeepMET, there is one output, and it is accessed through:

```cpp
  const auto& output1 = iOutput.begin()->second;
```

This can also be done in a similar way as the inputs:

```cpp
  const auto& output1 = iOutput.at("output/BiasAdd");
```

The results are saved as `vector<vector<float>>`, where the first dimension is the batch, and the 2nd dimension is the result per batch. In the DeepMET case, there is only one batch, and in each batch there is a 2-dimensional vector [here](https://github.com/fastmachinelearning/cmssw/blob/CMSSW\_12\_0\_0\_pre5\_SONIC/RecoMET/METPUSubtraction/plugins/DeepMETSonicProducer.cc#L118-L120):

```cpp
// outputs are px and py
float px = outputs[0][0] * norm_;
float py = outputs[0][1] * norm_;
```

Other postprocessings can be done with these inference outputs.

#### Python config

The last needed part is a python configuration to config some client settings and inputs. For DeepMET, the [python config](https://github.com/fastmachinelearning/cmssw/blob/CMSSW\_12\_0\_0\_pre5\_SONIC/RecoMET/METPUSubtraction/python/deepMETSonicProducer\_cff.py) is

```python
deepMETSonicProducer = cms.EDProducer("DeepMETSonicProducer",
    Client = cms.PSet(
        timeout = cms.untracked.uint32(300),
        mode = cms.string("Async"),
        modelName = cms.string("deepmet"),
        modelConfigPath = cms.FileInPath("HeterogeneousCore/SonicTriton/data/models/deepmet/config.pbtxt"),
        # version "1" is the resolutionTune
        # version "2" is the responeTune
        modelVersion = cms.string("1"),
        verbose = cms.untracked.bool(False),
        allowedTries = cms.untracked.uint32(0),
        useSharedMemory = cms.untracked.bool(True),
        compression = cms.untracked.string(""),
    ),
    pf_src = cms.InputTag("packedPFCandidates"),
)
```

Details of the meanings and options can be found [here](https://github.com/cms-sw/cmssw/tree/master/HeterogeneousCore/SonicTriton#client).

#### Test Script

A short test script to run the DeepMET SONIC producer is available [here](https://github.com/fastmachinelearning/cmssw/blob/CMSSW\_12\_0\_0\_pre5\_SONIC/RecoMET/METPUSubtraction/test/testDeepMETSonic\_cfg.py).&#x20;
