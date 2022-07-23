def all_finished(processes):
	for p in processes:
		is_running = processes[p].poll() is None 
		if is_running:
			return False
	return True