from decision_tree import *
import numpy as np


# Load the datasets
filepath = 'wifi_db/'
filename = 'clean_dataset.txt'
data = np.loadtxt(filepath+filename)

np.random.shuffle(data)

x_train = data[:1500, :-1]
y_train = data[:1500, -1]

x_test = data[1500:, :-1]
y_test = data[1500:, -1]


def evaluate(predicted_label, true_label):
    """
    Evaluate accuracy on predicted label given true label

    Args:
            predicted_label : 
            true_label      :
    Outputs: accuracy of the predicted label
    """
    assert(len(predicted_label) == len(true_label))
    data_num = len(true_label)
    accuracy = float(sum(predicted_label == true_label))/data_num
    return accuracy


def cross_validate(train_val_data, k_fold=10):
    """
    perform k-fold cross validation on learn_decision_tree
	iterable

    Args:
            train_val_data (np.array): train and val data with label on the last column
            k	 (int): data.shape[0]>=k>0
    Outputs:
			tree (class tree): the trained decision tree on each fold
			confusino matrix: the confusion matrix on each validation split
    """
    assert(type(train_val_data) == np.ndarray)
    assert(len(train_val_data.shape) == 2 and train_val_data.shape[1] > 1)

    num_row = train_val_data.shape[0]
    assert(type(k_fold) == int)
    assert(k_fold > 0 and k_fold <= num_row)

    np.random.shuffle(train_val_data)

    # compute the index of a split of train_val_data
    # def k_range(k_head, k_tail): 
    #     head=int(num_row*k_head/k_fold)
    #     tail=int(num_row*k_tail/k_fold)
    #     return np.arange(head, tail).astype(int)
    data_splits=np.vsplit(train_val_data,k_fold)

    for k in range(0, k_fold):
        # divide data to train/val set
        val_data = data_splits[k]
        if k==0:
            train_data = data_splits[0]
        else:
            train_data_splits=data_splits[0:k]+data_splits[k+1:k_fold]
            train_data = np.concatenate(train_data_splits)

        # train decision tree
        tree = Decision_tree()
        tree.train(train_data[:, :-1], train_data[:, -1])
        tree.prune(train_data[:, :-1], train_data[:, -1])

        yield tree,compute_confusion_matrix(tree, val_data, len(set(val_data[:, -1])))


def compute_confusion_matrix(tree, test_data, num_label):
    """
    compute confusion matrix of a learnt tree

    Args:
            tree	 (dict): learnt decision tree
            test_data (int): test data with label on the last column
            num_label (int): number of label category in dataset(row/col number of matrix)
    Outputs:
			confusion_matrix(np.ndarray,shape(num_label,num_label))
    """
    # assert(type(tree)==dict)
    assert(type(test_data) == np.ndarray)
    assert(len(test_data.shape) == 2 and test_data.shape[1] > 1)
    assert(type(num_label) == int)
    assert(num_label > 0)

    confusion_matrix = np.zeros([num_label, num_label])
    predicted_label = tree.classify(test_data[:, :-1])
    gt_label = test_data[:, -1]

    for p, g in zip(predicted_label, gt_label):
        confusion_matrix[int(p)-1, int(g)-1] += 1

    return confusion_matrix


def main():
	tree = Decision_tree()
	tree.train(x_train, y_train)

	y_pred = tree.classify(x_test)
	print("Accuracy: " + str(evaluate(y_pred, y_test)))

	tree.draw()

	tree.prune(x_test, y_test)
	y_pred = tree.classify(x_test)
	print("Accuracy: " + str(evaluate(y_pred, y_test)))

	confusion_matrix = compute_confusion_matrix(tree, data, len(set(y_test)))
	print(confusion_matrix)

	tree.draw()

if __name__=="__main__":
#def main2():
	for _,confusion_matrix in cross_validate(data,5):
		print(np.trace(confusion_matrix)/np.sum(confusion_matrix))


