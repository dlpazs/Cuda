# Tensor RT

A platform for deploying DL models onto the GPU deployable to embedded devices and onto datacenters.
It increases throughput and reducing latency during inference. TensorRt provides APIs and parses to import trained models from all
frameworks. 

## A Simple TensorRT Example

This example has 3 steps: importing a pretrained model into TensorRT, applying optimizations and generating an engine, and performing inference of the GPU.

- Firstly, loading a model and converting it into a TensorRT network. ONNX is  a standard for representing DL models enabling them to be transferred between frameworks, Caffe2, CNTK, Pytorch, MXNet support ONNX format). Next an optimised TensorRT engine is built based on the input model, target GPU platform and other configuration parameters specified. The last step is to provide data to the TensorRT engine to perform inference. 
- ONNX parser: Takes trained model in ONNX format as input and populates a network object in TensorRT
- Builder: takes a network in TensorRT and generates an engine that is optimized for the target platform
- Engine: takes input data, performs inference and outputs inference
- Logger: object associated with the builder and engine to capture errors, warnings and other info during build and inference

