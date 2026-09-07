#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/finetune_MultiMAE_anomaly.sh"
bash "${SCRIPT_DIR}/finetune_MultiMAE_capacity_EV.sh"
bash "${SCRIPT_DIR}/finetune_MultiMAE_capacity_storage.sh"
bash "${SCRIPT_DIR}/finetune_MultiMAE_capacity_lab.sh"
bash "${SCRIPT_DIR}/finetune_MultiMAE_RUL.sh"
bash "${SCRIPT_DIR}/finetune_MultiMAE_IR.sh"
