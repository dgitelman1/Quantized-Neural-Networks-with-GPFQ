# Quantized_Neural_Nets

## Oct. 21th, 2021

### Yixuan's Update

Set up the pipeline of training `MLP`. 

TODO: Currently the accuracy depends highly on radius, need figure out how to deal with it.

## Oct. 17th, 2021

### Yixuan's Update

Set up a general framework of quantizing neural networks.
The idea is we should have a runner that runs instantiate `QuantizeNeuralNet` 
that is defined in `quantize_neural_net.py` to start the quantization process.
After that, it should call `quantize_network` on the `QuantizeNeuralNet` 
instantiated to perform the quantization, which return a reference of the 
network that is quantized.
