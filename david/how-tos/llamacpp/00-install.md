# Install Llama Cpp

```bash
# pip uninstall llama-cpp-python
# source: https://github.com/abetlen/llama-cpp-python/issues/1617

CMAKE_ARGS="-DGGML_CUDA=on" LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu"  pip install llama-cpp-python
dpkg -S libcuda.so.1 
```



