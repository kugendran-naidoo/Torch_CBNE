export graph="Graph-2"
export file="../../../quantinuum/CBNE/graphs/${graph}.graphml"
ground_truth=$(grep "<data key='v_betti'>" ${file} | cut -d ">" -f 2 | cut -d "<" -f 1)

printf "\nCalibrating file: ${file}\n"
printf "Ground truth : ${ground_truth}\n\n"

time python calibrate_cbne.py \
  --path ../../../quantinuum/CBNE/graphs/${graph}.graphml \
  --target ${ground_truth} \
  --epsilon 0.1 \
  --deg-limit -1 3 \
  --iter-start 3000 \
  --iter-max 20000 \
  --iter-factor 1.5 \
  --seeds 123 \
  --tolerance 5e-4 \
  --min-improvement 2.5e-4 \
  --device cpu \
  --summary-json logs/calibration/${graph}_summary.json

printf "\nGround truth : ${ground_truth}\n\n"
