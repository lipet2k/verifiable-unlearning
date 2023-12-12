from flask import Flask, request, jsonify
from flask_cors import cross_origin, CORS
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
CORS(app)

output_dir = Path("/root/verifiable-unlearning/outputs")
working_dir = setup_working_dir(output_dir, str(uuid.uuid4()), overwrite=False)
proof_config = {
    "circ_path": Path('/root/circ'),
    "proof_system": "snark",
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

sisa = SISA(proof_config)
X, Y, scaler = Dataset.from_pmlb_shard('analcatdata_creditscore')
sisa.train(X, Y)

global_proof_params = []


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
        prediction, list_pred = sisa.predict(data, scaler)
        return jsonify({'prediction': prediction, 'list_pred': list_pred }), 200
    else:
        return "Method not allowed", 405

@app.route('/unlearn', methods=['POST'])
@cross_origin()
def unlearn():
    model_id_1 = request.args.get('model_id_1')
    datapoint_idx_1 = request.args.get('datapoint_idx_1')

    ids = [
        {'model_id': model_id_1, 'datapoint_idx': datapoint_idx_1},
    ]
    
    proof_params = sisa.unlearn(ids)

    try:
        shutil.copytree('/root/verifiable-unlearning/templates/poseidon', working_dir.joinpath('poseidon'))
    except:
        pass

    for proof_param in proof_params:
        proof_src = proof_param['proof']
        working_dir.joinpath('circuit.zok').write_text(proof_src)
        
    global global_proof_params
    global_proof_params += proof_params

    return jsonify(proof_params), 200


@app.route('/reset', methods=['POST'])
@cross_origin()
def reset():
    global working_dir
    working_dir = setup_working_dir(output_dir, str(uuid.uuid4()), overwrite=False)
    proof_config = {
        "circ_path": Path('/root/circ'),
        "proof_system": "snark",
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
    global sisa
    sisa = SISA(proof_config)
    X, Y, scaler = Dataset.from_pmlb_shard('analcatdata_creditscore')
    sisa.train(X, Y)
    
    global global_proof_params
    global_proof_params = []

    return jsonify("Reset successful"), 200

@app.route('/verify')
@cross_origin()
def verify():
    if request.method == 'GET':

        global global_proof_params
        for proof_param in global_proof_params:
            params = proof_param['params']
            try:
                circ = CirC(proof_config['circ_path'], debug=proof_config['debug'])
                stdout = circ.spartan_nizk(params, working_dir)
                return stdout, 200
            except:
                return "Verification failed", 400
    else:
        return "Method not allowed", 405
    
# @app.route('/verify_one')
# @cross_origin()
# def verify_one():
#     if request.method == 'GET':

#         global global_proof_params
#         if len(global_proof_params) == 0:
#             return "Verification Not Available", 400
#         proof_param = global_proof_params[0]
#         params = proof_param['params']
#         proof_src = proof_param['proof']

#         working_dir.joinpath('circuit.zok').write_text(proof_src)
#         circ = CirC(proof_config['circ_path'], debug=proof_config['debug'])
#         shutil.copytree('/root/verifiable-unlearning/templates/poseidon', working_dir.joinpath('poseidon'))
#         stdout = circ.spartan_nizk(params, working_dir)
#         return stdout, 200

#     else:
#         return "Method not allowed", 405

@app.route('/examples')
@cross_origin()
def examples():
    if request.method == 'GET':
        return sisa.examples(), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)