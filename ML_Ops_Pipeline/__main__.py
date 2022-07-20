if __name__ == "__main__":
	import traceback
	import argparse
	import time

	import torch

	from .main_git import main


	torch.multiprocessing.set_start_method('spawn')

	parser = argparse.ArgumentParser()
	parser.add_argument('--single', action='store_true', help='Run the loop only once')
	parser.add_argument('--disable-torch-multiprocessing', action='store_true', help='Disable multiprocessing')
	args = parser.parse_args()

	if args.single:
		main(disable_torch_multiprocessing=args.disable_torch_multiprocessing)
		exit()
		
	while True:
		try:
			main(disable_torch_multiprocessing=args.disable_torch_multiprocessing)
			time.sleep(5)
		except Exception as e:
			traceback.print_exc()
			print("Exception: {}".format(e))
			time.sleep(1)
