import numpy as np
import matplotlib.pyplot as plt

# Define essential functions for training a decision tree
def entropy(label):
	"""Compute the entropy given a list of labels
	Args:
	  label (list of integers or numpy array with size (n,)): A list of labels
	Returns:
	  entropy (float): enropy of the given list of labels
	"""
	assert (type(label) == np.ndarray or type(label) == list)

	labels = set(label)
	label_num = len(label)
	instances = dict.fromkeys(labels,0)
	for l in label:
		instances[l] += 1
	prob = [float(k)/label_num for k in list(instances.values())]
	sep_entropy = [p*np.log2(p) for p in prob]
	return -sum(sep_entropy)

#get the depth of each node
def get_node_num(node_):
	"""
	Usage: n = get_node_num(node)
	Description: Get the number of descendents of the node
	Input: a node which from class Node
	Output: number of descendents 
	"""
	num_leafs = 0
	if node_.left == None and node_.right == None:
		num_leafs += 1
	if node_.left != None:
		num_leafs += get_node_num(node_.left)
	if node_.right != None:
		num_leafs += get_node_num(node_.right)
	return num_leafs

def get_node_depth(node_):
	"""
	Usage: n = get_node_depth(node)
	Description: Get the depth of the node
	Input: a node which from class Node
	Output: The depth of the node 
	"""
	max_depth = 0
	if node_.attr == 'leaf':
		max_depth = 1
		return max_depth 
	else:
		if node_.left != None:
			this_depth_l = 1+get_node_depth(node_.left)
		else:
			this_depth_l = 1
		if node_.right != None:
			this_depth_r = 1+get_node_depth(node_.right)
		else:
			this_depth_r = 1
		if this_depth_l >= this_depth_r:
			max_depth = this_depth_l
		else:
			max_depth = this_depth_r
		return max_depth




def draw_node(fig, node_, x, d, depth, total_leaf):
	"""
	Usage: x,d = draw_node(node_, x, d, depth, total_leaf)
	Description: Get the position of node base on its parent's position and depth and the number of descentdents
	Input: a node which from class Node, position of parent, the depth of parent, and the tolal number of leaf in this tree
	Output: the position of this node and the depth of this node 
	"""
	node_num = get_node_num(node_)
	node_depth = get_node_depth(node_)
	move = 1+node_num*node_depth
	if node_.attr == "leaf":
		plt.text(x, d, str(node_.value), fontsize= 10)
		return x, d
	else:
		x_l, d_l = draw_node(fig, node_.left, x+move, d-1, depth, total_leaf)
		x_r, d_r = draw_node(fig, node_.right, x-move, d-1, depth, total_leaf)
		plt.plot([x, x_l], [d, d_l], 'b')
		plt.scatter([x, x_l], [d, d_l], color = 'r', s = 10)
		plt.plot([x, x_r], [d, d_r], 'b')
		plt.scatter([x_r], [d_r], color = 'r', s = 10)
		plt.text(x, d, str(node_.attr)+"<"+str(node_.value),fontsize = 10)
		if d == depth:
			plt.axis('off')
			return fig
		else:
			return x, d


# Define the decision tree class
class Node():
	"""Node that serves as basic building blocks for the decision tree
	
	The node contains the neccesary attributes of splitting a decision tree.
	Node() class is also designed to enable pruning process to run smoothly.
	
	Attributes
		attr: The attribute of which to split a decision tree. The type would depend on the attributes
		value: An integer that indicates the value to split
		parent: A Node object which is the parent of this node
		left: A Node object which is the left child of this node
		right: A Node object which is the right child of this node
		data_count: A dictionary the maps the labels and the number of occurrences that a data with such label visited a tree. This is only used in pruning and would be set to None if no data had visited or pruning ended.

	Functions
		__init__(self): initialize attributes
		set_parent(self, node_): set current node's parent to node_
		add_child(self, node_, direction): set current node's child at given direction to node_
		set_attr(self, attr): set the node's attribute
		set_value(self, value): set the node's value
		clear_child(self): clear the node's child
		set_data_count(self, label_num): initialize the data_count dictionary
		update_data_count(self, label): update the data_count with the given label
		clear_visit_history(self): set data_count to None
	"""
	def __init__(self):
		"""initialize attributes"""
		self.attr = None
		self.value = None
		self.parent = None
		self.left = None
		self.right = None
		# The following variable are used for pruning
		self.data_count = None
	def set_parent(self, node_):
		"""set current node's parent to node_
		Args:
		  node_ (Node): the node that would be set to be the parent 
		"""
		assert (isinstance(node_, Node))
		self.parent = node_
	def add_child(self, node_, direction):
		"""set current node's child at given direction to node_
		Args:
		  node_       (Node): the node that would be set to be the child
		  direction (string): 'left' or 'right'
		"""
		assert (direction == 'left' or direction == 'right')
		assert (isinstance(node_, Node))
		if direction == "left":
			self.left = node_
		elif direction == "right":
			self.right = node_

	def set_attr(self, attr):
		"""set the node's attribute
		Args:
		  attr	: The attribute to be set
		"""
		self.attr = attr
	def set_value(self, value):
		"""set the node's value
		Args:
		  value (int or float): the value to be set
		"""
		assert (type(value) == int or type(value) == float or type(value) == np.float64)
		self.value = value
	def clear_child(self):
		"""clear the node's child"""
		self.left = None
		self.right = None
	def set_data_count(self, label_num):
		"""initialize the data_count dictionary
		Args:
		  label_num (set): A set of all possible labels
		"""
		self.data_count = dict.fromkeys(label_num,0)
	def update_data_count(self, label):
		"""update the data_count with the given label
		Args:
		  label : a label that should be an element of label_num used in set_data_count(label_num)
		"""
		assert(label in self.data_count.keys())
		self.data_count[label] += 1
	def clear_visit_history(self):
		"""set data_count to None"""
		self.data_count = None


class Decision_tree():
	"""Decision_tree that can be trained and pruned
	
	Attributes
		root  (Node): The root Node of the decision_tree (Initialized to None)
		depth  (int): The depth of the initially trained decision_tree (Initialized to 0). Does not change after pruning
		leafs (list of Nodes): A list of Nodes that is the leafs of the tree
	Functions
		__init__(self): initialize attributes
		find_split(self, data, label): find the best split given the data and label
		decision_tree_learning(data, label, d = 0): learn the decision tree recurrsively
		train(self, data, label): train the tree by calling decision_tree_learning
		classify(self, data): classify test data based on trained results
		prop(self, data, label): propogate and record data that passed through each node
		clear_prune_history(self, node_): clear the propogated data after pruning(recurrsive)
		prune(self, data, label): prune the tree with given data and label
		draw(self): Visualize the tree
	"""
	def __init__(self):
		"""Initialize attributes"""
		self.root = None
		self.depth = 0
		self.leafs = []
	def find_split(self, data, label):
		"""Find the best split attribute and value for the given data and label.
		Args:
		  data     (numpy array with size (n,num_of_attributes)): Data used for training
		  label (list of integers or numpy array with size (n,)): A list of labels
		Returns:
		  node_attr                         (int): The attribute used for splitting
		  node_split               (int or float): The value used for splitting
		  l_set	(tuple of the form:(data, label)): A tuple containing the data and labels that is splitted to the left
		  r_set	(tuple of the form:(data, label)): A tuple containing the data and labels that is splitted to the right
		""" 
		assert (type(data) == np.ndarray and len(data.shape) == 2)
		assert(type(label) == np.ndarray or type(label) == list)
		data_num, attr_num = data.shape

		info_gain = float('-inf')
		node_split = None
		node_attr = None
		l_set = np.empty([0, data.shape[1]])
		r_set = np.empty([0, data.shape[1]])
		for attr in range(attr_num):
			# Find the optimal split in this attribute
			data, label = (data[data[:,attr].argsort()], label[data[:,attr].argsort()])
			for i in range(data_num - 1):
				if data[i, attr] == data[i+1,attr]:
					continue
				else:
					split = (data[i+1, attr] + data[i,attr])/2.0
					#split the data based on the calculated split
					l_branch = data[:,attr] > split
					l_data, l_label = (data[l_branch], label[l_branch])
					r_branch = data[:,attr] <= split
					r_data, r_label = (data[r_branch], label[r_branch])
					# Calculate the information gain with this split
					# For simplicity we calculate the remainder and take the minus sign to quantify it
					H_l = entropy(l_label)
					H_r = entropy(r_label)
					neg_remainder = -float(len(l_label)*H_l + len(r_label)*H_r)/data_num
					if  neg_remainder > info_gain:
						info_gain = neg_remainder
						node_split = split
						node_attr = attr
						l_set = (l_data, l_label)
						r_set = (r_data, r_label)
		return node_attr, node_split, l_set, r_set

	def decision_tree_learning(self, data, label, d = 0):
		"""learn the decision tree recurrsively
		Args:
		  data     (numpy array with size (n,num_of_attributes)): Data used for training
		  label (list of integers or numpy array with size (n,)): A list of labels
		  d                                      (int default:0): The current depth of the tree
		Returns:
		  node_                        (Node): The splitted node
		  d               				(int): The current record of depth
		  leaf_l               (list of Node): A list of current leafs of the tree
		"""
		assert (type(data) == np.ndarray and len(data.shape) == 2)
		assert(type(label) == np.ndarray or type(label) == list)
		assert(type(d) == int)
		node_ = Node()
		label_num = len(set(label))
		if label_num == 1:
			node_.set_attr('leaf')
			node_.set_value(label[0])
			return node_, d, [node_]
		else:
			attr, split, l_set, r_set = self.find_split(data, label)
			node_.set_attr(attr)
			node_.set_value(split)
			node_l, d_l, leaf_l = self.decision_tree_learning(l_set[0], l_set[1], d+1)
			node_r, d_r, leaf_r = self.decision_tree_learning(r_set[0], r_set[1], d+1)
			node_l.set_parent(node_); node_r.set_parent(node_)
			node_.add_child(node_l, "left")
			node_.add_child(node_r, "right")			
			d = max(d_l, d_r)
			leaf_l.extend(leaf_r) 
			return node_, d, leaf_l

	def train(self, data, label):
		"""train the tree by calling decision_tree_learning
		Args:
		  data     (numpy array with size (n,num_of_attributes)): Data used for training
		  label (list of integers or numpy array with size (n,)): A list of labels
		"""
		assert (type(data) == np.ndarray and len(data.shape) == 2)
		assert(type(label) == np.ndarray or type(label) == list)
		self.root, self.depth, self.leafs = self.decision_tree_learning(data, label)

	def classify(self, data):
		"""classify test data based on trained results
		Args:
		  data (numpy array with size (n,num_of_attributes)): Data to be classified
		Returns:
		  label 							   (list of int): Predicted labels
		"""
		assert (type(data) == np.ndarray)
		label = []
		for d in data:
			tree = self.root
			while True:
				attr = tree.attr
				if attr == "leaf":
					label.append(tree.value)
					break
				else:
					split = tree.value
					if d[attr] > split:
						tree = tree.left
					else:
						tree = tree.right
		return label

	def prop(self, data, label):
		"""propogate and record data that passed through each node
		Args:
		  data     (numpy array with size (n,num_of_attributes)): Data to be propogated
		  label (list of integers or numpy array with size (n,)): The corresponding labels for the data 		
		"""
		assert (type(data) == np.ndarray and len(data.shape) == 2)
		assert(type(label) == np.ndarray or type(label) == list)
		label_num = set(label)
		# In a prop tree we use attr to count all labels passed through and use value to count the total visit number
		self.root.set_data_count(label_num)
		for d, label in zip(data, label):
			curr_tree = self.root
			while True:
				# Update the prop data
				curr_tree.update_data_count(label)
				if curr_tree.attr == 'leaf':
					break
				else:
					attr, split = (curr_tree.attr, curr_tree.value)
					if d[attr] > split:
						curr_tree = curr_tree.left
						if curr_tree.data_count == None: curr_tree.set_data_count(label_num)
					else:
						curr_tree = curr_tree.right
						if curr_tree.data_count == None: curr_tree.set_data_count(label_num)

	def clear_prune_history(self, node_):
		"""clear the propogated data after pruning(recurrsive)
		Args:
		  node_    (Node): The Node to be cleared. All descendants of this node would also be cleared.
		"""
		assert(isinstance(node_, Node))
		node_.clear_visit_history()
		if node_.attr == 'leaf':
			return
		else:
			self.clear_prune_history(node_.left)
			self.clear_prune_history(node_.right)

	def prune(self, data, label):
		"""prune the tree with given data and label
		Args:
		  data     (numpy array with size (n,num_of_attributes)): Data to be used for pruning
		  label (list of integers or numpy array with size (n,)): The corresponding labels for the data 
		"""
		assert (type(data) == np.ndarray and len(data.shape) == 2)
		assert (type(label) == np.ndarray or type(label) == list)
		self.prop(data, label)
		zero_dict = dict.fromkeys(set(label),0)
		leafs = self.leafs
		while True:
			new_leafs = []
			pruned = False
			for leaf in leafs:
				parent = leaf.parent
				if parent.left == None:
					#parent left == None checks if the children had been pruned
					continue
				elif parent == None or parent.left.attr != 'leaf' or parent.right.attr != 'leaf':
					new_leafs.append(leaf)
				else: #Now both sides of the parent should be leafs
					# If no testing data had passed through a leaf. we treat it as zero acc
					l_data = zero_dict if parent.left.data_count == None else parent.left.data_count
					r_data = zero_dict if parent.right.data_count == None else parent.right.data_count
					child_acc = float(l_data[int(parent.left.value)]+r_data[int(parent.right.value)])/max(sum(l_data.values())+sum(r_data.values()),1)
					p_data = [0] if parent.data_count == None else parent.data_count.values()
					parent_acc = float(max(p_data))/max(sum(p_data),1)
					# Check if we prune
					if parent_acc >= child_acc and parent_acc != 0:
						pruned = True
						parent.clear_child()
						new_val = list(parent.data_count.keys())[list(p_data).index(max(p_data))]
						parent.set_value(new_val)
						parent.set_attr('leaf')
						new_leafs.append(parent)
					else:
						new_leafs.append(leaf)
			leafs = new_leafs
			if not pruned:
				break
		self.leafs = leafs
		self.clear_prune_history(self.root)
	
	def draw(self, filename):
		"""
		Usage: draw(tree)
		Description: This function is to draw the tree
		Input: The tree
		Outputs: No output
		"""
		total_leaf = get_node_num(self.root)
		fig = plt.figure(figsize=(32,9))
		fig = draw_node(fig, self.root, 0, self.depth, self.depth, total_leaf)
		fig.savefig(filename)
		#plt.clf()
