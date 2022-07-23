import time
import os
import subprocess
import traceback
import signal
from .constants import LOG_DIR, DATA_BASE_DIR
from .helper import all_finished

os.makedirs(LOG_DIR, exist_ok=True)



try:
	process_list = [
		'model_utils/model_training', 
		'model_utils/model_analysis', 
		'model_utils/model_visualizer_loop', 
		# 'ensemble_utils/ensemble_training',
		# 'ensemble_utils/ensemble_analysis', 
		# 'ensemble_utils/ensemble_visualizer_loop'
	]
	processes = {}

	for process in process_list:
		stdout_process = open(os.path.join(LOG_DIR, "stdout_" + process + ".log"), 'w')
		stderr_process = open(os.path.join(LOG_DIR, "stderr_" + process + ".log"), 'w')
		process_py = process + '.py'
		assert os.path.exists(process_py)
		print("Starting ", process)
		processes[process] = subprocess.Popen(['/usr/bin/python', process_py], stdout=stdout_process, stderr=stderr_process)

	# browsepy 0.0.0.0 8080 --directory data
	process = 'browsepy'
	stdout_process = open(os.path.join(LOG_DIR, "stdout_" + process + ".log"), 'w')
	stderr_process = open(os.path.join(LOG_DIR, "stderr_" + process + ".log"), 'w')
	processes[process] = subprocess.Popen(['browsepy', '0.0.0.0', '8080', '--directory', DATA_BASE_DIR], stdout=stdout_process, stderr=stderr_process)

	# FLASK_APP=rest_server.py FLASK_ENV=development flask run
	process = 'rest_server'
	stdout_process = open(os.path.join(LOG_DIR, "stdout_" + process + ".log"), 'w')
	stderr_process = open(os.path.join(LOG_DIR, "stderr_" + process + ".log"), 'w')
	my_env = os.environ.copy()
	my_env['FLASK_APP'] = "rest_server.py"
	my_env['FLASK_ENV'] = "development"
	processes[process] = subprocess.Popen(['flask', 'run'], stdout=stdout_process, stderr=stderr_process, env=my_env)


	try:
		while True:
			print("I am alive")
			for process in processes:
				is_running = processes[process].poll() is None 
				if is_running:
					print("is_running: ", process)
				else:
					print("DOWN: ", process)
			time.sleep(10)
	except KeyboardInterrupt as ex:
		print("Shutdown signal recieved")
		#traceback.print_exc()
	finally:
		while not all_finished(processes):
			try:
				for process in processes:
					print("Sending shutdown signal to", process)
					processes[process].send_signal(signal.SIGINT)
				for process in processes:
					print("Waiting for ", process)
					processes[process].wait()
			except KeyboardInterrupt as ex:
				print("Please wait for all processes to halt")
				#traceback.print_exc()
			finally:
				pass
except Exception as ex:
	print("Critical error")
	traceback.print_exc()
finally:
	while not all_finished(processes):
		try:
			for process in processes:
				print("Sending shutdown signal to", process)
				processes[process].send_signal(signal.SIGINT)
			for process in processes:
				print("Waiting for ", process)
				processes[process].wait()
		except KeyboardInterrupt as ex:
			print("Please wait for all processes to halt")
			#traceback.print_exc()
		finally:
			pass
