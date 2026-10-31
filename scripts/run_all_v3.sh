#!/bin/bash
# Gemma 4 31B round (docs/v3_gemma/README.md): scripts/run_all.sh v3_gemma.
#   LOCUS_ROOT=/shared/project NODES=/shared/project/work/v3_gemma/nodes.txt scripts/run_all_v3.sh
exec "$(dirname "$0")/run_all.sh" v3_gemma
