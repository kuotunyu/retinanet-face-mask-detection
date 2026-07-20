# Environment Notes

This project uses TensorFlow 1.13.2 and Keras 2.1.5, so reproducibility depends on an older Python/CUDA stack.

## Recommended Conda Setup

```bash
conda env create -f environment.yml
conda activate retinanet
```

For the optional Gradio demo:

```bash
pip install -r requirements-demo.txt
```

## CPU-only Setup

For CPU-only usage, replace:

```text
tensorflow_gpu==1.13.2
```

with:

```text
tensorflow==1.13.2
```

Then install with:

```bash
pip install -r requirements.txt
```

## Docker

The included `Dockerfile` is intended to document a reproducible CUDA 10.0 / cuDNN 7 runtime path for this legacy stack.

```bash
docker build -t retinanet-mask .
```

Run a command inside the environment:

```bash
docker run --rm -it retinanet-mask python predict.py --help
```

For GPU execution, use NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all -it retinanet-mask python demo.py --server-name 0.0.0.0 --server-port 7860
```

## Notes

- Python 3.6 is end-of-life, so use an isolated environment.
- TensorFlow 1.x GPU builds are sensitive to CUDA/cuDNN versions.
- The smoke tests in `tests/` do not load TensorFlow or model weights; they are meant for quick repository sanity checks.
