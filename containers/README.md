# Containers (Apptainer / Docker)

## Why use a container here?

On this HPC cluster, **vLLM** is hard to use outside a container:

- Login nodes are often **ARM**; vLLM has no prebuilt ARM wheels, so `pip install vllm` tries to build from source and can fail (e.g. compiler `-march` errors).
- **Apptainer** (HPC-friendly, no root) lets you run a **pre-built image** that was built elsewhere (e.g. on x86_64 with CUDA). That image already has vLLM and the right stack, so you don’t build vLLM on the cluster.

So “build an image” means: **create a container image (with Docker or Apptainer) that has Python, CUDA, vLLM, and your project deps**. You build it once (on a machine with Docker, or with Apptainer from a definition file), then on the cluster you **run** that image. Your code and data stay on the cluster; you typically **bind-mount** the repo and data into the container when you run it.

## Quick start

### Option A: Build an image with Apptainer (on the cluster or a build node)

If your cluster allows `apptainer build` (many do):

```bash
cd /projects/bfgx/jparekh/query-conditioned-guidelines
apptainer build qcg.sif containers/apptainer.def
```

Then run training or a shell:

```bash
# Bind-mount project and data; run a shell so you can run scripts
apptainer exec --nv --bind .:/workspace/query-conditioned-guidelines qcg.sif bash
# inside container:
cd /workspace/query-conditioned-guidelines
export PYTHONPATH=.
./experiments/run_train_20step_test.sh
```

`--nv` enables NVIDIA GPUs; `--bind` mounts the repo so the container sees your code and data.

### Option B: Build from the Dockerfile (on a machine with Docker), then use on HPC

On a machine that has Docker (e.g. your laptop or a build node):

```bash
cd /projects/bfgx/jparekh/query-conditioned-guidelines
docker build -f containers/Dockerfile -t qcg:latest .
```

Transfer the image to the cluster (e.g. save/load or push to a registry the cluster can pull from), then convert to Apptainer or run with Docker if allowed:

```bash
# If the cluster can pull from a registry:
apptainer build qcg.sif docker://your-registry/qcg:latest

# Or on a machine with both Docker and Apptainer, build then convert:
docker save qcg:latest -o qcg.tar
# copy qcg.tar to cluster, then:
apptainer build qcg.sif docker-archive:qcg.tar
```

Run the same way as Option A (`apptainer exec --nv --bind ...`).

### Option C: Use an official vLLM image and mount your code

You can skip building a custom image and use an official vLLM + CUDA image, then mount the repo and install your deps at run time (or add a small wrapper script). Example:

```bash
apptainer exec --nv --bind .:/workspace/query-conditioned-guidelines \
  docker://nvcr.io/nvidia/pytorch:24.01-py3 \
  bash -c "pip install vllm math-verify mathruler && cd /workspace/query-conditioned-guidelines && PYTHONPATH=. ./experiments/run_train_20step_test.sh"
```

(Adjust the image tag and install list as needed; the first run will be slower due to pip installs.)

## What’s in this directory

| File | Purpose |
|------|--------|
| `README.md` | This file: what “build an image” means and how to build/run. |
| `Dockerfile` | Image with CUDA, Python, vLLM, and project pip deps. Project code is mounted at run time. |
| `apptainer.def` | Apptainer definition to build the same stack without Docker (build with `apptainer build`). |

## Running training / eval inside the container

After starting a shell in the image (e.g. `apptainer exec ... bash`):

- **Data prep** (run once):  
  `python -m experiments.prepare_numinamath --output_dir experiments/data/numinamath_30k`
- **20-step test**:  
  `./experiments/run_train_20step_test.sh`
- **Full training**:  
  `./experiments/run_train.sh`
- **MATH-500 eval**:  
  `PYTHONPATH=. python -m experiments.run_eval_math500 --model Qwen/Qwen2.5-3B-Instruct`

Make sure the repo is bind-mounted to the same path you use inside the container (e.g. `/workspace/query-conditioned-guidelines`), and that `PYTHONPATH` includes the project root when running Python.

## Slurm / batch jobs

Run the container inside your job script, e.g.:

```bash
#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH ...
apptainer exec --nv --bind .:/workspace/query-conditioned-guidelines /path/to/qcg.sif \
  bash -c "cd /workspace/query-conditioned-guidelines && ./experiments/run_train.sh"
```

Use the same `--nv` and `--bind` as above.
