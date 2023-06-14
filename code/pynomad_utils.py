import sys
sys.path.append("/nomad-v.4.3.1/interfaces/PyNomad")
import PyNomad, random, os, signal
from utils import std_dev

function_obj = None
num_points = None


def run_nomad():
	global function_obj, num_points
	# init first evaluation point
	x0 = []
	if not function_obj.has_input_domain_range:
		lb = []
		ub = []
		for d in range(function_obj.dimension):
			x0.append(random.uniform(-100,100))
	else:
		lb = function_obj.input_lb
		ub = function_obj.input_ub
		for d in range(function_obj.dimension):
			x0.append(random.uniform(function_obj.input_lb[d], function_obj.input_ub[d]))

	params = ["BB_OUTPUT_TYPE OBJ", "MAX_BB_EVAL 1000000 ", 
			f"DIMENSION {function_obj.dimension}", 
			"DISPLAY_DEGREE 0", 
			"DISPLAY_UNSUCCESSFUL False", 
			f"BB_MAX_BLOCK_SIZE {num_points}"
			]
	# perform optimization
	nomad_res = PyNomad.optimize(bb_block, x0, lb, ub, params)
	nomad_res = {"num evaluations" : nomad_res["nb_evals"], "input opt" : nomad_res["x_best"], "opt" : nomad_res["f_best"]}
	return nomad_res


def bb_block(block):
	# evaluate input points inside block
	global function_obj
	nbPoints = block.size()
	evalOk = [False for i in range(nbPoints)]
	try:
		for k in range(nbPoints):
			x = block.get_x(k)
			inp = [x.get_coord(i) for i in range(x.size())]
			f = function_obj.evaluate(inp)
			x.setBBO(str(f).encode('UTF-8'))
			evalOk[k] = True
	except Exception as e:
		print("Unexpected error in bb_block()", str(e))
		return evalOk
	return evalOk