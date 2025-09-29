#!/usr/bin/env zsh
set -euo pipefail

graph="Graph-2"
base_dir="$(cd "$(dirname "$0")" && pwd)"
project_root="${base_dir}/../.."
repo_root="${project_root}/.."
venv="${project_root}/venv_tcbne/bin/activate"

if [[ -f "${venv}" ]]; then
  source "${venv}"
else
  printf "Warning: virtualenv not found at %s\n" "${venv}"
fi

file="${repo_root}/sample_graphs/quantinuum/${graph}.graphml"
if [[ ! -f "${file}" ]]; then
  printf "Graph file not found: %s\n" "${file}" >&2
  exit 1
fi

ground_truth=$(grep "<data key='v_betti'>" "${file}" | cut -d ">" -f 2 | cut -d "<" -f 1)

printf "\nCalibrating %s with extended sweep\n" "${graph}"
printf "Graph file   : %s\n" "${file}"
printf "Ground truth : %s\n\n" "${ground_truth}"

log_dir="${base_dir}/logs/calibration"
mkdir -p "${log_dir}"
summary_json="${log_dir}/${graph}_extended_summary.json"

cd "${base_dir}" || exit 1

time python calibrate_cbne.py \
  --path "${file}" \
  --target "${ground_truth}" \
  --epsilon 0.05 0.1 0.2 \
  --deg-limit -1 6 8 \
  --iter-start 4000 \
  --iter-max 80000 \
  --iter-factor 1.25 \
  --seeds 123 456 789 987 \
  --tolerance 3e-4 \
  --min-improvement 1e-4 \
  --max-stalled 5 \
  --device cpu \
  --summary-json "${summary_json}" \
  --log-dir "${log_dir}/${graph}"

printf "\nCalibration complete. Summary: %s\n" "${summary_json}"
printf "Ground truth : %s\n\n" "${ground_truth}"
