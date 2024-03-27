from .helper import *
import numpy as np
import scipy
from typing import Set, List, Tuple
import torch
import torch.nn as nn
import random
from ..SetFunction import SetFunction

class DisparitySumFunction(SetFunction):

	def __init__(self, n, mode, sijs=None, data=None, metric="cosine", num_neighbors=None,batch_size=0):
		super(DisparitySumFunction, self).__init__()
		self.n = n
		self.mode = mode
		self.metric = metric
		self.sijs = sijs
		self.data = data
		self.num_neighbors = num_neighbors
		self.cpp_obj = None
		self.cpp_sijs = None
		self.cpp_content = None
		self.effective_ground_set = None
		self.batch_size = batch_size
		if self.n <= 0:
			raise Exception("ERROR: Number of elements in ground set must be positive")

		if self.mode not in ['dense', 'sparse']:
			raise Exception("ERROR: Incorrect mode. Must be one of 'dense' or 'sparse'")

		# print(self.sijs)
		# if self.metric not in ['euclidean', 'cosine']:
		# 	raise Exception("ERROR: Unsupported metric. Must be 'euclidean' or 'cosine'")

		if type(self.sijs) != type(None): # User has provided similarity kernel
			if type(self.sijs) == scipy.sparse.csr.csr_matrix:
				if num_neighbors is None or num_neighbors <= 0:
					raise Exception("ERROR: Positive num_neighbors must be provided for given sparse kernel")
				if mode != "sparse":
					raise Exception("ERROR: Sparse kernel provided, but mode is not sparse")
			elif type(self.sijs) == np.ndarray:
				if mode != "dense":
					raise Exception("ERROR: Dense kernel provided, but mode is not dense")
			else:
				raise Exception("Invalid kernel provided")
			#TODO: is the below dimensionality check valid for both dense and sparse kernels?
			if np.shape(self.sijs)[0]!=self.n or np.shape(self.sijs)[1]!=self.n:
				raise Exception("ERROR: Inconsistentcy between n and dimensionality of given similarity kernel")
			if type(self.data) != type(None):
				print("WARNING: similarity kernel found. Provided data matrix will be ignored.")

		#it goes here
		else: #similarity kernel has not been provided
			if type(self.data) != type(None):
				if np.shape(self.data)[0]!=self.n:
					raise Exception("ERROR: Inconsistentcy between n and no of examples in the given data matrix")
				# print(self.sijs)
				if self.mode == "dense":
					if self.num_neighbors  is not None:
						raise Exception("num_neighbors wrongly provided for dense mode")
					self.num_neighbors = np.shape(self.data)[0] #Using all data as num_neighbors in case of dense mode



				self.data = map(np.ndarray.tolist,self.data)
				self.data = list(self.data)

				y = torch.tensor(self.data)
				z = y.to(device="cuda")
				x= create_kernel(X=z, metric=self.metric, num_neigh=self.num_neighbors, mode=self.mode, batch = self.batch_size)
				# print("type x",type(x))
				#self.cpp_content = np.array(create_kernel(X=torch.tensor(self.data, device="cuda").cpu(), metric=self.metric, num_neigh=self.num_neighbors, mode=self.mode).to_dense())
				#val = self.cpp_content[0]
				val = x[0].cpu().detach().numpy()
				#print("val type", val.shape)
				row = list(x[1].cpu().detach().numpy().astype(int))
				#row = list(self.cpp_content[1].astype(int))
				#print("row type", type(row))
				col = list(x[2].cpu().detach().numpy().astype(int))
				if self.mode=="dense":
					self.sijs = np.zeros((n,n))
					#print(self.sijs.shape)
					self.sijs[row,col] = val
				if self.mode=="sparse":
					self.num_neighbors = 0
					self.sijs = scipy.sparse.csr_matrix((val, (row, col)), [n,n])
			else:
				raise Exception("ERROR: Neither ground set data matrix nor similarity kernel provided")

		cpp_ground_sub = {-1} #Provide a dummy set for pybind11 binding to be successful

		#Breaking similarity matrix to simpler native data structures for implicit pybind11 binding
		if self.mode=="dense":
			#print(self.sijs.shape)
			self.cpp_sijs = self.sijs.tolist() #break numpy ndarray to native list of list datastructure
			#print(type(self.cpp_sijs[0][0]))
			#print(self.cpp_sijs)
			if type(self.cpp_sijs[0])==int or type(self.cpp_sijs[0])==float: #Its critical that we pass a list of list to pybind11
																			 #This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
				l=[]
				l.append(self.cpp_sijs)
				self.cpp_sijs=l
				#print(type(len(self.cpp_sijs)))

			# self.cpp_obj = DisparityMin(self.n, self.cpp_sijs, False, cpp_ground_sub)
			#                             	n          sijs     partial   ground

			self.effective_ground_set = set(range(n))
			self.numeffectivegroundset  = len(self.effective_ground_set)
			self.currentSum = 0



		if self.mode=="sparse": #break scipy sparse matrix to native component lists (for csr implementation)
			self.cpp_sijs = {}
			self.cpp_sijs['arr_val'] = self.sijs.data.tolist() #contains non-zero values in matrix (row major traversal)
			self.cpp_sijs['arr_count'] = self.sijs.indptr.tolist() #cumulitive count of non-zero elements upto but not including current row
			self.cpp_sijs['arr_col'] = self.sijs.indices.tolist() #contains col index corrosponding to non-zero values in arr_val
			if(len(self.cpp_sijs['arr_val']) ==0 or len(self.cpp_sijs['arr_count']) ==0 or len(self.cpp_sijs['arr_col']) ==0):
				raise Exception("Error: Empty/Corrupt sparse similarity kernel")
			self.sparse_kernel = subcp.SparseSim(self.cpp_sijs['arr_val'],self.cpp_sijs['arr_count'],self.cpp_sijs['arr_col'])
			self.effective_ground_set = set(range(n))
			self.numeffectivegroundset = len(self.effective_ground_set)
			self.currentSum = 0

	def evaluate(self, X: Set[int]) -> float:
		effective_X = X
		if len(effective_X) == 0 :
			return 0.0
		if self.mode == 'dense':
			return get_sum_dense(effective_X, self)
		elif self.mode == 'sparse':
			return get_sum_sparse(effective_X, self)
		else:
			raise ValueError("Error: Only dense and sparse mode supported")

	def evaluate_with_memoization(self, X: Set[int]) -> float:
		return self.currentSum

	def get_effective_ground_set(self) -> Set[int]:
		return self.effective_ground_set

	def marginal_gain(self, X: Set[int], item: int) -> float:
			effective_X = X
			gain = 0.0

			if item in effective_X:
					return 0.0

			if item not in self.effective_ground_set:
					return 0.0


			if self.mode == 'dense':
					for elem in effective_X:
							gain += (1 - self.cpp_sijs[elem][item])
			elif self.mode == 'sparse':
					for elem in effective_X:
							gain += (1 - self.sparse_kernel.get_val(item, elem))
			else:
					raise ValueError("Error: Only dense and sparse mode supported")

			return gain

	def update_memoization(self, X: Set[int], item: int) -> None:
			self.currentSum += self.marginal_gain(X, item)

	def clear_memoization(self) -> None:
			self.currentSum = 0.0

	def set_memoization(self, X: Set[int]) -> None:
			self.currentSum = self.evaluate(X)
