# Sourced by every launch/ script. Paths are taken from the environment, never hard-coded:
#   LOCUS_ROOT  shared project directory, the same path on every node (required)
#   LCAM_CODE   this repository's checkout (default: the directory above launch/)
#   PY          python of the training venv (default $LOCUS_ROOT/venv/bin/python)
#   BUNDLE      data bundle (default $LOCUS_ROOT/data/locus-bundle)
#   LCAM_MODEL  local Qwen3.5-35B-A3B weights (default $LOCUS_ROOT/models/Qwen3.5-35B-A3B)
: "${LOCUS_ROOT:?set LOCUS_ROOT to the shared project directory}"
LCAM_CODE=${LCAM_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${PY:-$LOCUS_ROOT/venv/bin/python}
BUNDLE=${BUNDLE:-$LOCUS_ROOT/data/locus-bundle}
export LCAM_MODEL=${LCAM_MODEL:-$LOCUS_ROOT/models/Qwen3.5-35B-A3B}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
SELF_IP=${SELF_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}
