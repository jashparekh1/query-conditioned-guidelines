#!/bin/bash
# One-time installation of flash-attn in verl_env
# Run this inside the container to install flash-attn once

CONTAINER=/projects/bfgx/jparekh/test_ngc/torch2501.sif

echo "Installing flash-attn==2.4.3.post1 in verl_env..."
singularity exec --nv \
  --bind /projects/bfgx/jparekh:/projects/bfgx/jparekh \
  $CONTAINER bash -c "
    cd /projects/bfgx/jparekh/test_ngc
    source verl_env/bin/activate
    MAX_JOBS=4 pip install --no-build-isolation flash-attn==2.4.3.post1
    echo 'Flash-attn installation complete!'
    pip show flash-attn
"

