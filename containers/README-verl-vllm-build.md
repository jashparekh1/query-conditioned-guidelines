# Build verl + vLLM image for GH200 (aarch64, driver 550, CUDA 12.4)

Base image: `nvcr.io/nvidia/pytorch:24.04-py3` (CUDA 12.4; requires driver 545+, you have 550).

You must build **on aarch64** (e.g. a Grace Hopper compute node). Building on x86 will produce an image that cannot run on GH200.

---

## Step 1: Create build directory on scratch

On the cluster (login or a node):

```bash
mkdir -p /work/nvme/bfgx/jparekh/verl-vllm-build
cd /work/nvme/bfgx/jparekh/verl-vllm-build
```

---

## Step 2: Copy the Apptainer definition file into the build dir

Apptainer does **not** accept a Dockerfile directly (it parses it as a .def and fails). Use the `.def` file:

```bash
cp /projects/bfgx/jparekh/query-conditioned-guidelines/containers/verl-vllm.def \
   /work/nvme/bfgx/jparekh/verl-vllm-build/
```

---

## Step 3: Get a build environment on aarch64

You need either **Docker** or **Apptainer** that can build from a Dockerfile, on an **aarch64** machine.

### Option A: Docker on a compute node (if allowed)

Some clusters allow Docker on a build node or in a partition. Check with support. Then:

```bash
cd /work/nvme/bfgx/jparekh/verl-vllm-build
docker pull nvcr.io/nvidia/pytorch:24.04-py3
docker build -t verl-vllm:24.04 .
```

Then either push to a registry and pull with Apptainer, or convert to SIF (Step 4B).

### Option B: Apptainer build from definition file (recommended if no Docker)

On a **compute node** (so the image is built for aarch64), use the **.def** file (not the Dockerfile—Apptainer does not parse Dockerfiles):

```bash
srun --account=bfgx-dtai-gh --partition=ghx4 --nodes=1 --gpus=0 --time=1:00:00 --pty bash
# once on the node:
cd /work/nvme/bfgx/jparekh/verl-vllm-build
apptainer build --fakeroot verl-vllm-24.04.sif verl-vllm.def
```

If `--fakeroot` is not allowed, try without it (may require root or sandbox):

```bash
apptainer build verl-vllm-24.04.sif verl-vllm.def
```

### Option C: Build on another aarch64 machine (e.g. cloud)

If you have an aarch64 server or cloud instance (e.g. AWS Graviton) with Docker:

```bash
docker build -t your-registry/verl-vllm:24.04 .
docker push your-registry/verl-vllm:24.04
```

On the cluster (login node):

```bash
cd /work/nvme/bfgx/jparekh
apptainer build verl-vllm-24.04.sif docker://your-registry/verl-vllm:24.04
```

---

## Step 4: Place the final SIF where your scripts expect it

- Your current scripts use: `SIF="$WORK_NVME/verl_v1-arm-flashinfer.sif"` with `WORK_NVME=/work/nvme/bfgx/jparekh`.
- So put the new image there and point the scripts at it.

Either:

**A) Replace the old SIF (after testing):**

```bash
mv /work/nvme/bfgx/jparekh/verl-vllm-build/verl-vllm-24.04.sif \
   /work/nvme/bfgx/jparekh/verl_v1-arm-flashinfer.sif
```

(Back up the old one first if you want to keep it.)

**B) Or keep both and change the script to the new name:**

In `slurm/run_train_20step_test.slurm` and `slurm/run_train_20step_interactive.sh` and `enter_apptainer.sh`, set:

```bash
SIF="$WORK_NVME/verl-vllm-24.04.sif"
```

and copy the SIF to that path:

```bash
cp /work/nvme/bfgx/jparekh/verl-vllm-build/verl-vllm-24.04.sif \
   /work/nvme/bfgx/jparekh/verl-vllm-24.04.sif
```

---

## Step 5: Run your 20-step test

Same as before; no need to use `/nvme/verl` if verl is inside the image (optional: keep using `/nvme/verl` from setup_verl_apptainer.sh or rely on the image’s `/opt/verl` and set `VERL_PATH=/opt/verl` in the script).

If the image has verl at `/opt/verl`, you can set in your run scripts:

- `VERL_PATH=/opt/verl` (and no need to run setup_verl_apptainer.sh to clone verl; you can still source it for HF_* and TRITON_* if you want).

Then:

```bash
cd /projects/bfgx/jparekh/query-conditioned-guidelines
sbatch slurm/run_train_20step_test.slurm
```

---

## Troubleshooting

- **“Cannot build from Dockerfile”**: Use an Apptainer definition file (`.def`) that uses `Bootstrap: docker` and `From: nvcr.io/nvidia/pytorch:24.04-py3`, then in `%post` run the same `pip` and `git` steps. I can write that `.def` if you need it.
- **Wrong architecture**: If the NGC image pulls as amd64 on your login node, run the pull (and build) from an **aarch64 compute node** so the correct image is used.
- **pip install vllm fails**: Build on a node with GPU or enough RAM; or use a prebuilt vLLM wheel for aarch64 if your cluster provides one.
