import numpy as np
import matplotlib.pyplot as plt

# Define essential functions for training a decision tree
def entropy(label):
	"""
	Usage: H = entropy(label)
	Description: This function calculates the entropy given a data set assuming the last column represents the labels
	Input: 2d numpy array
	Output: single value representing the entropy
	"""
	
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
	num_leafs = 0
	if node_.left == None and node_.right == None:
		num_leafs += 1
	if node_.left != None:
		num_leafs += get_node_num(node_.left)
	if node_.right != None:
		num_leafs += get_node_num(node_.right)
	return num_leafs

def get_node_depth(node_):
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




def draw_node(node_, x, d, depth, total_leaf):
	node_num = get_node_num(node_)
	node_depth = get_node_depth(node_)
	move = 1+node_num*node_depth
	if node_.attr == "leaf":
		plt.text(x, d, str(node_.value), fontsize= 10)
		return x, d
	else:
		x_l, d_l = draw_node(node_.left, x+move, d-1, depth, total_leaf)
		x_r, d_r = draw_node(node_.right, x-move, d-1, depth, total_leaf)
		plt.plot([x, x_l], [d, d_l], 'b')
		plt.scatter([x, x_l], [d, d_l], color = 'r', s = 10)
		plt.plot([x, x_r], [d, d_r], 'b')
		plt.scatter([x_r], [d_r], color = 'r', s = 10)
		plt.text(x, d, str(node_.attr)+"<"+str(node_.value),fontsize = 10)
		if d == depth:
			plt.axis('off')
			plt.show()
		else:
			return x, d


# Define the decision tree class
class Node():
	def __init__(self):
		self.attr = None
		self.value = None
		self.parent = None
		self.left = None
		self.right = None
		# The following variable are used for pruning
		self.data_count = None
	def set_parent(self, node_):
		self.parent = node_
	def add_child(self, node_, direction):
		if direction == "left":
			self.left = node_
		elif direction == "right":
			self.right = node_
		else:
			print("The direction of the child should be either left or right.")
	def set_attr(self, attr):
		self.attr = attr
	def set_value(self, value):
		self.value = value
	def clear_child(self):
		self.left = None
		self.right = None
	def set_data_count(self, label_num):
		self.data_count = dict.fromkeys(label_num,0)
	def update_data_count(self, label):
		self.data_count[label] += 1
	def clear_visit_history(self):
		self.data_count = None


class Decision_tree():
	def __init__(self):
		self.root = None
		self.depth = 0
		self.leafs = []
	def find_split(self, data, label):
		"""
		Usage:
		Description:
		Outputs:
		"""
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
		self.root, self.depth, self.leafs = self.decision_tree_learning(data, label)

	def classify(self, data):
		label = []
		for d in data:
			tree = self.root
			at_leaf = False
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

	def create_prop_node(self, label_num, parent=None):
		p_node = Node()
		p_node.set_attr(dict.fromkeys(label_num,0))
		p_node.set_value(0)
		p_node.set_parent(parent)
		return p_node

	def prop(self, data, label):
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
		node_.clear_visit_history()
		if node_.attr == 'leaf':
			return
		else:
			self.clear_prune_history(node_.left)
			self.clear_prune_history(node_.right)

	def prune(self, data, label):
		self.prop(data, label)
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
					l_data = [0] if parent.left.data_count == None else parent.left.data_count.values()
					r_data = [0] if parent.right.data_count == None else parent.right.data_count.values()
					child_acc = float(max(l_data)+max(r_data))/max(sum(l_data)+sum(r_data),1)
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
	
	def draw(self):
		total_leaf = get_node_num(self.root)
		draw_node(self.root, 0, self.depth, self.depth, total_leaf)
