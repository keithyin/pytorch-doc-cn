# 多线程最佳实践

`torch.multiprocessing` is a drop in replacement for Python’s multiprocessing module. It supports the exact same operations, but extends it, so that all tensors sent through a `multiprocessing.Queue`, will have their data moved into shared memory and will only send a handle to another process.
