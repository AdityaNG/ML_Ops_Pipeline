import os
import pickle
from datetime import datetime
from .constants import HISTORY_PATH

class local_history(dict):

	def __init__(self, name) -> None:
		self.name = name
		os.makedirs(HISTORY_PATH, exist_ok=True)
		self.local_history_pkl = os.path.join(HISTORY_PATH, self.name + ".pkl")
		self.load()


	def load(self) -> object:
		if os.path.exists(self.local_history_pkl):
			local_history_handle = open(self.local_history_pkl, 'rb')
			self.dict = pickle.load(local_history_handle)
			local_history_handle.close()
		else:
			self.dict = {}

	def store(self) -> object:
		local_history_handle = open(self.local_history_pkl, 'wb')
		pickle.dump(self.dict, local_history_handle, protocol=pickle.HIGHEST_PROTOCOL)
		local_history_handle.close()

	def __setitem__(self, key, item):
		self.dict[key] = item
		self.store()

	def __getitem__(self, key):
		self.load()
		if not self.has_key(key):
			self.__setitem__(key, datetime.fromtimestamp(0))
		return self.dict[key]

	def __repr__(self):
		return repr(self.dict)

	def __len__(self):
		return len(self.dict)

	def __delitem__(self, key):
		del self.dict[key]

	def clear(self):
		return self.dict.clear()

	def copy(self):
		return self.dict.copy()

	def has_key(self, k):
		return k in self.dict

	def update(self, *args, **kwargs):
		return self.dict.update(*args, **kwargs)

	def keys(self):
		return self.dict.keys()

	def values(self):
		return self.dict.values()
