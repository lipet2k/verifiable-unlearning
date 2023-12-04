set -x
set -e

source ~/.bashrc

/usr/bin/python3 /root/verifiable-unlearning/src/run.py \
    --trial_name models/neural_network_2 \
    --technique retraining --mode train \
    --no_samples_D_prev 0 \
    --no_samples_D_plus 1 \
    --no_samples_U_prev 0 \
    --no_samples_U_plus 0 \
    --dataset_name analcatdata_creditscore \
    --classifier neural_network_2 \
    --proof_system nizk