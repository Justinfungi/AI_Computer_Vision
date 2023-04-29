# AI_ComputerVision


### Bugs

RuntimeError: CUDA out of memory. Tried to allocate 74.00 MiB (GPU 0; 10.76 GiB total capacity; 9.47 GiB already allocated; 62.44 MiB free; 9.83 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

- The RuntimeError: CUDA out of memory error occurs when the GPU runs out of memory while training a neural network. This can happen if the model, input data, or batch size is too large for the available GPU memory.

- To resolve this issue, you can try one of the following steps:

- Reduce the batch size: The batch size determines the number of input samples that are processed at once. If the batch size is too large, it may not fit into the GPU memory. Try reducing the batch size to a smaller value.

- Use smaller models: If the model is too large, it may not fit into the GPU memory. You can try using a smaller model or reducing the number of layers in the model.

- Free up memory: If you have other Python processes running on the GPU or if there are cached GPU tensors that you are no longer using, they may be using up GPU memory. Try clearing the cache by restarting the kernel or freeing up memory before running your code.

- Upgrade your GPU: If none of the above steps work, you may need to upgrade your GPU to one with more memory.
