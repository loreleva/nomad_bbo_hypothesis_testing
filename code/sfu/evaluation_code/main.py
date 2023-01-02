import sys, time
from functions import *

function_obj = None

def eucl_dist(x, y):
	sum_diff = 0
	for idx in range(len(x)):
		sum_diff += (y[idx] - x[idx])**2
	return math.sqrt(sum_diff)

def get_dist(algo_input):
	global function_obj
	if function_obj.minimum_x == None:
		return None
	else:
		if type(function_obj.minimum_x[0]) != list:
			res = eucl_dist(algo_input, function_obj.minimum_x)
			return res
		else:
			results = []
			for inp in function_obj.minimum_x:
				results.append(eucl_dist(algo_input, inp))
			res = min(results)
			return res

if __name__ == "__main__":
	#print(f"Result search: {search_function({'minimum_f' : True})}")
	#print("Functions with dimension d:{}\n".format(search_function({"dimension" : "d"})))
	function_selected = "branin_function"
	function_obj = objective_function(function_selected)
	print(get_dist([8.14159,2.275]))
	
	