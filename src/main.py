from flask import Flask, request, jsonify
from flask_cors import cross_origin
from sisa import SISA
from dataset import Dataset
from copy import deepcopy
from circ import CirC
import json
import log_setup
import uuid
from pathlib import Path
from utils import setup_working_dir
from loguru import logger as log
import shutil

app = Flask(__name__)

perm_sisa = SISA()
X, Y = Dataset.from_pmlb_shard('analcatdata_creditscore')
perm_sisa.train(X, Y)

sisa = deepcopy(perm_sisa)

global_proof_params = []

output_dir = Path("/root/verifiable-unlearning/outputs")
working_dir = setup_working_dir(output_dir, str(uuid.uuid4()), overwrite=False)
proof_config = {
    "circ_path": Path('/root/circ'),
    "proof_system": "nizk",
    "circuit_dir": Path('/root/verifiable-unlearning/templates'),
    "epochs": 1,
    "lr": 0.01,
    "classifier": "neural_network_2",
    "precision": 1e5,
    "debug": False,
    "model_seed": 2023,
    "unlearning_epochs": 1,
    "unlearining_lr": 0.01,
    "working_dir": working_dir,
}

@app.route('/classify')
@cross_origin()
def classify():
    if request.method == 'GET':
        age = request.args.get('age')
        income = request.args.get('income')
        credit_ex = request.args.get('credit_ex')
        home_ownership = request.args.get('home_ownership')
        employment = request.args.get('employment')
        reports = request.args.get('reports')

        data = [age, income, credit_ex, home_ownership, employment, reports]
        data = [float(x) for x in data]
        prediction, list_pred = sisa.predict(data)
        return jsonify({'prediction': prediction, 'list_pred': list_pred }), 200
    else:
        return "Method not allowed", 405

@app.route('/unlearn', methods=['POST'])
@cross_origin()
def unlearn():
    if request.method == 'POST':
        model_id_1 = request.form.get('model_id_1')
        datapoint_idx_1 = request.form.get('datapoint_idx_1')

        ids = [
            {'model_id': model_id_1, 'datapoint_idx': datapoint_idx_1},
        ]
        
        proof_params = sisa.unlearn(ids)

        global global_proof_params
        global_proof_params = proof_params

        return jsonify(proof_params), 200
    else:
        return "Method not allowed", 405

@app.route('/reset', methods=['POST'])
@cross_origin()
def reset():
    global sisa
    sisa = deepcopy(perm_sisa)

@app.route('/verify', methods=["POST"])
@cross_origin()
def verify():
    if request.method == 'POST':
        proof_src = request.form.get('proof_src')
        working_dir.joinpath('circuit.zok').write_text(proof_src)
        circ = CirC(proof_config['circ_path'], debug=proof_config['debug'])
        shutil.copytree('/root/verifiable-unlearning/templates/poseidon', working_dir.joinpath('poseidon'))

        global global_proof_params
        for proof_param in global_proof_params:
            params = proof_param['params']
            try:
                stdout = circ.spartan_nizk(params, working_dir)
                return stdout, 200
            except:
                pass

        return "Verification failed", 400

    else:
        return "Method not allowed", 405

@app.route('/examples')
@cross_origin()
def examples():
    if request.method == 'GET':
        return sisa.examples(), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)