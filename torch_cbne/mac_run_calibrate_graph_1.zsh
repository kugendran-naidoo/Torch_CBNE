#!/usr/bin/env zsh
set -euo pipefail

graph="Graph-1"
base_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="${base_dir}/../../.."
venv="${repo_root}/venv_tcbne/bin/activate"

if [[ -f "${venv}" ]]; then
  source "${venv}"
else
  printf "Warning: virtualenv not found at %s\n" "${venv}"
fi

file="${repo_root}/quantinuum/CBNE/graphs/${graph}.graphml"
if [[ ! -f "${file}" ]]; then
  printf "Graph file not found: %s\n" "${file}" >&2
  exit 1
fi

ground_truth=$(grep "<data key='v_betti'>" "${file}" | cut -d ">" -f 2 | cut -d "<" -f 1)

printf "\nCalibrating ${graph} with enhanced sweep\n"
printf "Graph file   : %s\n" "${file}"
printf "Ground truth : %s\n\n" "${ground_truth}"

log_dir="${base_dir}/logs/calibration"
mkdir -p "${log_dir}"
summary_json="${log_dir}/${graph}_enhanced_summary.json"

cd "${base_dir}" || exit 1

time python calibrate_cbne.py \
  --path "${file}" \
  --target "${ground_truth}" \
  --epsilon 0.1 \
  --deg-limit -1 3 \
  --iter-start 3000 \
  --iter-max 20000 \
  --iter-factor 1.5 \
  --seeds 123 \
  --tolerance 5e-4 \
  --min-improvement 2.5e-4 \
  --device cpu \
  --summary-json "${summary_json}" \
  --log-dir "${log_dir}/${graph}"

printf "\nCalibration complete. Summary: %s\n" "${summary_json}"
printf "Ground truth : %s\n\n" "${ground_truth}"
