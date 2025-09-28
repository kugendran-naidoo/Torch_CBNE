cd /Volumes/PData/Data/Users/kn/Data/Dev/Github/Repos/phd3/cbne/local
source venv_tcbne/bin/activate
pip install -e Torch_CBNE            # once per environment
time python Torch_CBNE/scripts/run_cbne_logged.py \
     --path ../quantinuum/CBNE/graphs/Graph-1.graphml \
     --iter_limit 5000 \
     --deg_limit 3 \
     --device cpu \
     --seed 123 \
     --log logs/cbne_iter5000.log
