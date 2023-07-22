import time, sys, os, math, itertools, random
from memory_profiler import memory_usage
from utils import *
import write_logs as log
import pynomad_utils

# execute N run of nomad, in each run we have num_workers parallel evaluation of the function, then we take the minim evaluation among num_workers as temp result of nomad


num_proc = 0
num_points = 0
function_obj = None
q_inp = None
q_res = None
path_dir_log_file = None
csv_sep = ";"



def multiproc_function(q_inp, q_res, function_obj, num_points):
	"""Function runned by the concurrent processes. Each process executes a run of nomad if
	it can pop from the queue "q_inp" the value 1. If the popped value is None, then it stops.
	After each execution of nomad, each process push in the queue "q_res" a dictionary with the results of the execution, which has the keys:
	- "process time" : process time in seconds of the execution
	- "max ram usage" : maximum ram usage during the run
	- "num evaluations" : number of objective function evaluations
	- "input_opt" : input values of the objective function that nomad thinks are the ones producing the optimum
	- "opt" : optimum found by nomad
	"""
	inp = q_inp.get()
	while (inp != None):
		# init time and ram tracking variables
		start_time = time.process_time()
		# run nomad
		pynomad_utils.function_obj = function_obj
		pynomad_utils.range_stopping_criteria = 20
		pynomad_utils.num_points = num_points
		res = pynomad_utils.run_nomad()
		# return results
		mem_usage = memory_usage(-1, max_usage=True)
		res.update({"process time" : time.process_time() - start_time, "max ram usage" : max(mem_usage)})
		q_res.put(res)
		# wait for a new run
		inp = q_inp.get()

def hypothesis_testing(input_opt, opt, delta, epsilon, path_dir_log_file, num_proc, q_inp, q_res, tolerance = 1e-6, correct_thr = 1e-6, verbose=False):
	# init N
	N = math.ceil((math.log(delta)/math.log(1-epsilon)))

	# track execution time of algorithm
	start_time = time.time()
	
	# init log files
	log.init_log_files(path_dir_log_file)

	if verbose:
		print(f"N: {N}")
	
	# start insert N+1 (the +1 is to obtain the first value of S) requests of runs of Nomad
	for _ in range(N+1):
		q_inp.put(1)

	# retrieve the result of a run
	res = q_res.get()
	
	# list of objective function errors of each run of nomad
	obj_function_errors = [objective_function_error(opt, res["opt"])]
	# list of runtime of each run of nomad
	processes_runs_time = [res["process time"]]
	# list of number of function evaluations of each run of nomad
	number_of_asks = [res["num evaluations"]]
	# list of max ram usage of each run of nomad
	ram_usage = [res["max ram usage"]]
	# keep the max ram usage among all the runs of nomad
	max_ram_usage = res["max ram usage"]
	# number of nomad runs executed
	num_run = 1

	# write log of the first run
	log.write_single_run_log(path_dir_log_file, res, num_run, 0, N, opt, input_opt, "", verbose) 
	
	# init S and S prime
	# S_values contains tuple (run number, input_opt, opt)
	S_values = [{"run number" : 1, 
				"internal iteration" : 0,
				"S value" : res["opt"],
				"input" : res["input opt"]
				}
	]
	S_prime = res["opt"]

	# keep the count of the correct results, where a solution is correct if obj function error is <= correct_thr
	num_correct_opts = 0
	if objective_function_error(opt, res["opt"]) <= correct_thr:
		num_correct_opts += 1
	
	num_external_iterations = 1
	while (1):
		S = S_prime

		# keep the count of the internal iterations
		internal_iter = 0

		# start the N iterations
		while(internal_iter < N):
			internal_iter += 1
			num_run += 1

			# obtain results and add it the lists
			res = q_res.get()
			# add objective function error
			obj_function_errors.append(objective_function_error(opt, res["opt"]))
			# add execution time
			processes_runs_time.append(res["process time"])
			# add number of function evaluations
			number_of_asks.append(res["num evaluations"])
			# add max ram usage of the run
			ram_usage.append(res["max ram usage"])
			# if new max among all the runs, update max ram usage
			if max_ram_usage < res["max ram usage"]:
				max_ram_usage = res["max ram usage"]

			# write log run
			log.write_single_run_log(path_dir_log_file, res, num_run, internal_iter, N, opt, input_opt, S, verbose) 

			# count correct results
			if objective_function_error(opt, res["opt"]) <= correct_thr:
				num_correct_opts += 1

			# if result smaller than S, restart
			if (res["opt"] + tolerance) < S:
				S_prime = res["opt"]
				S_values.append({"run number" : num_run, 
								"internal iteration" : internal_iter,
								"S value" : res["opt"],
								"input" : res["input opt"]
								}
				)
				break

		# if after the N iterations no better S is found, end the algorithm
		if S == S_prime:
			break

		num_external_iterations += 1

		# before restarting the N runs, add a number of requests in order to execute N runs
		for _ in range(internal_iter):
			q_inp.put(1)

	# WRITE LOGS OF THE RESULTS

	# write final result log
	log.write_final_result_log(
			path_dir_log_file,
			num_run,
			num_external_iterations,
			S_values[-1]["S value"],
			opt,
			S_values[-1]["input"],
			input_opt,
			num_correct_opts/num_run,
			time.time() - start_time,
			processes_runs_time,
			num_proc,
			number_of_asks,
			ram_usage,
			max_ram_usage,
			verbose
		)


	# write log for table of S values
	log.write_s_values_log(
			path_dir_log_file,
			S_values,
			opt,
			input_opt,
			N,
			verbose
		)
	

	# write log for plot of S values
	log.write_s_values_plot_log(
			path_dir_log_file,
			S_values,
			num_run
		)


	# write log correctness ratio i.e., shows the behaviour of the number of correct results during the runs, according to 5 different values for correctness ratio
	# obtain dictionary with correctness ratio values as keys and the corresponding error threshold as value
	log.write_correctness_ratio_plot_log(
			path_dir_log_file,
			obj_function_errors,
			num_run
		)


	# write log for plot of the values of correctness ratio for different value of threshold
	log.write_correctness_ratios_plot_log(
						path_dir_log_file, 
						obj_function_errors,
						num_run
					)


	# write log for plot of execution times
	log.write_time_percentage_plot(path_dir_log_file, processes_runs_time)


	# write log for plot of ram usage
	log.write_ram_percentage_plot(path_dir_log_file, ram_usage)


	# write log for plot of num function evals
	log.write_f_evals_percentage_plot(path_dir_log_file, number_of_asks)


	# write log errors plot
	log.write_error_plot(path_dir_log_file, obj_function_errors)