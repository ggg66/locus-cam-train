# Sourced by every launch/ script. Paths are taken from the environment, never hard-coded:
#   LOCUS_ROOT  shared project directory, the same path on every node (required)
#   LCAM_CODE   this repository's checkout (default: the directory above launch/)
#   PY          python of the training venv (default $LOCUS_ROOT/venv/bin/python)
#   BUNDLE      data bundle (default $LOCUS_ROOT/data/locus-bundle)
#   LCAM_ROUND  training round: configs/<round>/, work/<round>/, work/runs/<round>/, work/outputs/<round>/
#               (default v2 = Qwen3.5-35B-A3B; v3_gemma = Gemma 4 31B, docs/v3_gemma/README.md; v4 = supervision study on the v2 base, docs/v4/README.md)
#   LCAM_MODEL  local weights of the round's base model (default: Qwen3.5-35B-A3B for v2, gemma-4-31B-it for v3_gemma)
: "${LOCUS_ROOT:?set LOCUS_ROOT to the shared project directory}"
LCAM_CODE=${LCAM_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${PY:-$LOCUS_ROOT/venv/bin/python}
BUNDLE=${BUNDLE:-$LOCUS_ROOT/data/locus-bundle}
export LCAM_ROUND=${LCAM_ROUND:-v2}
if [ "$LCAM_ROUND" = v3_gemma ]; then export LCAM_MODEL=${LCAM_MODEL:-$LOCUS_ROOT/models/gemma-4-31B-it}
else export LCAM_MODEL=${LCAM_MODEL:-$LOCUS_ROOT/models/Qwen3.5-35B-A3B}; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
SELF_IP=${SELF_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}
