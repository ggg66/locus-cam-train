#!/bin/bash
# Supervision study on the v2 base (docs/v4/README.md): scripts/run_all.sh v4.
#   LOCUS_ROOT=/shared/project NODES=/shared/project/work/v4/nodes.txt ADDON_TAR=... ADDON_SHA256=... scripts/run_all_v4.sh
exec "$(dirname "$0")/run_all.sh" v4
