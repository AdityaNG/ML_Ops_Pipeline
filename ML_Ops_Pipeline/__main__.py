if __name__ == "__main__":
	import traceback
	import argparse
	import time

	import os
	import subprocess
	from multiprocessing import Process
	import traceback
	import signal

	import torch
	import mlflow

	from .main_git import main
	from .constants import LOG_DIR, MLFLOW_DIR, DATA_BASE_DIR
	from .helper import all_finished


	torch.multiprocessing.set_start_method('spawn')

	parser = argparse.ArgumentParser()
	parser.add_argument('--single', action='store_true', help='Run the loop only once')
	parser.add_argument('--disable-torch-multiprocessing', action='store_true', help='Disable multiprocessing')
	args = parser.parse_args()

	mlflow.set_tracking_uri("file://" + MLFLOW_DIR)


	if args.single:
		main(disable_torch_multiprocessing=args.disable_torch_multiprocessing)
		exit()
	else:
		try:
			processes = {}

			# mlflow ui --host 0.0.0.0 --backend-store-uri file://...
			process = 'mlflow'
			stdout_process = open(os.path.join(LOG_DIR, "stdout_" + process + ".log"), 'w')
			stderr_process = open(os.path.join(LOG_DIR, "stderr_" + process + ".log"), 'w')
			processes[process] = subprocess.Popen(['mlflow', 'ui', '--host', '0.0.0.0', '--backend-store-uri', 'file://' + MLFLOW_DIR], stdout=stdout_process, stderr=stderr_process)

			# python -m ML_Ops_Pipeline.main_git
			process = 'main_git'
			stdout_process = open(os.path.join(LOG_DIR, "stdout_" + process + ".log"), 'w')
			stderr_process = open(os.path.join(LOG_DIR, "stderr_" + process + ".log"), 'w')
			#processes[process] = subprocess.Popen(['python', '-m', 'ML_Ops_Pipeline.main_git', '--disable-torch-multiprocessing'], stdout=stdout_process, stderr=stderr_process)
			processes[process] = subprocess.Popen(['python', '-m', 'ML_Ops_Pipeline.main_git', '--disable-torch-multiprocessing'])

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
