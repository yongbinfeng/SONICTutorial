---
description: Collection of some useful relevant paper, document, and code links for SONIC
---

# Useful Links

### Papers:

* CMS Workflow demonstration with as a service: [arxiv:2402.15366](https://arxiv.org/abs/2402.15366)
* GPU coprocessors as a service for HEP: [arxiv.2007.10359](https://arxiv.org/abs/2007.10359)
* GPU coprocessors as a service for neutrinos: [arxiv.2009.04509](https://arxiv.org/abs/2009.04509)
* FPGAs as a service toolkit (FaaST): [arxiv.2010.08556](https://arxiv.org/abs/2010.08556)
* FPGA inference as a service: [arxiv.1904.08986](https://arxiv.org/pdf/1904.08986.pdf)

### Documents:

* Some introductory slides for beginners: [Link](files/SONIC\_Introduction\_MLHATS.pdf)
* NVIDIA Triton Inference server document: [Link](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton\_inference\_server\_230/user-guide/docs/)
* Distributed Triton documents with more details and latest developments: [Link](https://github.com/triton-inference-server/server/tree/main/docs)
* SONIC Core module in CMSSW: [Link](https://github.com/cms-sw/cmssw/tree/master/HeterogeneousCore/SonicCore)
* SONIC Triton module in CMSSW: [Link](https://github.com/cms-sw/cmssw/tree/master/HeterogeneousCore/SonicTriton)

### Code

* `cmsTriton` script to start the Triton server with Singularity in CMSSW: [Link](https://github.com/cms-sw/cmssw/blob/master/HeterogeneousCore/SonicTriton/scripts/cmsTriton)
* SONIC Producer examples and python configurations: [Link](https://github.com/cms-sw/cmssw/tree/master/HeterogeneousCore/SonicTriton/test)
* Some SONIC model examples in the central repo: [Link](https://github.com/fastmachinelearning/sonic-models/tree/master/models), which currently includes FACILE, ParticleNet, DeepTau, DeepMET, etc.
* Some SONC model producers that have been recently developed:
  * ParticleNet: [Link](https://github.com/fastmachinelearning/cmssw/blob/CMSSW\_12\_0\_0\_pre5\_SONIC/RecoBTag/ONNXRuntime/plugins/ParticleNetSonicJetTagsProducer.cc)
  * DeepTau: [Link](https://github.com/fastmachinelearning/cmssw/blob/CMSSW\_12\_0\_0\_pre5\_SONIC/RecoTauTag/RecoTau/plugins/DeepTauIdSonicProducer.cc)
  * DeepMET: [Link](https://github.com/fastmachinelearning/cmssw/blob/CMSSW\_12\_0\_0\_pre5\_SONIC/RecoMET/METPUSubtraction/plugins/DeepMETSonicProducer.cc)
