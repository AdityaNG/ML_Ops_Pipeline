import subprocess
import os
import importlib
import traceback
import glob

from constants import REMOTE_PIPELINES_DIR, REMOTE_PIPELINES_TXT

import git

log = False
if __name__=="__main__":
    log = True

global all_inputs, all_inputs_modules
all_inputs = {}
all_inputs_modules = {}

# def git_clone(repo_url, output_dir):
# 	git_clone_res = subprocess.check_output(['git', 'clone', repo_url, output_dir])
# 	pass

# def git_pull(working_dir):
# 	git_pull_res = subprocess.check_output(['git', 'pull'], cwd=working_dir)
# 	pass

# def git_latest_commit(working_dir):
# 	git_latest_commit_res = subprocess.check_output(['git', 'log', '-n', '1'], cwd=working_dir)
# 	return git_latest_commit_res


def get_all_inputs():
	global all_inputs, all_inputs_modules

	os.makedirs(REMOTE_PIPELINES_DIR, exist_ok=True)
	all_latest_commits = {}

	git_list_file = open(REMOTE_PIPELINES_TXT, 'r')
	pipelines_git_list = []
	for line in git_list_file.readlines():
		pipeline_git = line.replace('\n', '')
		pipelines_git_list.append(pipeline_git)

	for pipeline_git in pipelines_git_list:
		pipeline_name = pipeline_git.split('.git')[-2].split('/')[-1]
		pipeline_dir = os.path.join(REMOTE_PIPELINES_DIR, pipeline_name)
		if os.path.exists(pipeline_dir):
			repo = git.Repo(pipeline_dir)	
		else:
			print("Cloning repo: ", pipeline_git)
			repo = git.Repo.clone_from(pipeline_git, pipeline_dir)
		
		#origin = repo.remote
		origin = repo.remotes.origin
		assert origin.__class__ is git.remote.Remote
		assert not repo.bare
		assert repo.__class__ is git.Repo

		origin.pull()


		latest_commit = repo.head.commit #git_latest_commit(pipeline_dir)'
		"""
		latest_commit: 
		 'author', 'author_tz_offset', 'authored_date', 'authored_datetime', 'binsha', 
		 'committed_date', 'committed_datetime', 'committer', 'committer_tz_offset', 
		 'conf_encoding', 'count', 'create_from_tree', 'data_stream', 'default_encoding', 
		 'diff', 'encoding', 'env_author_date', 'env_committer_date', 'gpgsig', 'hexsha', 
		 'iter_items', 'iter_parents', 'list_items', 'list_traverse', 'message', 'name_rev', 
		 'new', 'new_from_sha', 'parents', 'replace', 'repo', 'size', 'stats', 'stream_data', 
		 'summary', 'trailers', 'traverse', 'tree', 'type']
		"""
		for diff_added in latest_commit.diff('HEAD~1').iter_change_type('A'):
			print(diff_added)

		all_latest_commits[pipeline_name] = latest_commit
		#all_latest_commits[pipeline_name] = latest_hash.decode('UTF-8')

	git_list_file.close()

	pipelines_list = list(map(lambda x: x.split("/")[-1],glob.glob(os.path.join(REMOTE_PIPELINES_DIR, '*'))))
	#pipelines_list = list(map(lambda x: x.split("/")[-1],glob.glob('remote_pipelines/*')))
	pipelines_list = list(filter(lambda x: x not in ["template", "__pycache__", "README.md"], pipelines_list))

	if log:
		print(pipelines_list)
		print("-"*10)

	for p in pipelines_list:
		try:
			p_name = p.split('/')[-1]
			if p_name in all_inputs:
				pipeline = importlib.reload(all_inputs_modules[p_name])
				if log:
					print("Reloading:", p_name)
			else:
				pipeline = importlib.import_module(REMOTE_PIPELINES_DIR + "." + p_name)
				if log:
					print("Loading:", p_name)
				
			p_exported = pipeline.exported_pipeline
			all_inputs[p_name] = p_exported
			all_inputs_modules[p_name] = pipeline
		except Exception as e:
			if log:
				print("FAILED: ", p, e)
				traceback.print_exc()
				
			pass
		finally:
			if log:
				print("-"*10)
	
	print(all_inputs)
	print(all_latest_commits)

	final_pipelines = {}
	for pipeline_name in all_inputs:
		final_pipelines[pipeline_name] = {
			'pipeline': all_inputs[pipeline_name],
			'git_data': all_latest_commits[pipeline_name]
		}
	return final_pipelines

if __name__=="__main__":
	pipelines_dict = get_all_inputs()
	for pipeline_name in pipelines_dict:
		print("-"*10)
		print(pipeline_name)
		print(pipelines_dict[pipeline_name]['pipeline'])
		print(pipelines_dict[pipeline_name]['git_data'])