import decision_tree as dt
import numpy as np
import sys


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


def cross_validate(model, train_val_data, k_fold=10, compare_prune = False):
    """
    perform k-fold cross validation on decision tree
	iterable

    Args:
            train_val_data (np.array): train and val data with label on the last column
            k	 (int): data.shape[0]>=k>0
            compare_prune  (bool): whether to compare the pruned tree
    Outputs:
			tree (class tree): the trained decision tree on each fold
			cm_dict (dict): the dictionary with {'unpruned':confusion_matrix, 'pruned':confusion_matrix(or None)}
    """
    assert(type(train_val_data) == np.ndarray)
    assert(len(train_val_data.shape) == 2 and train_val_data.shape[1] > 1)
    assert(type(compare_prune) == bool)

    num_row = train_val_data.shape[0]
    assert(type(k_fold) == int)
    assert(k_fold > 0 and k_fold <= num_row)

    np.random.shuffle(train_val_data)
    data_splits=np.vsplit(train_val_data,k_fold)

    for k in range(0, k_fold):
        # Initialize confusion matrix dict
        cm_dict = {'unpruned':None, 'pruned':None}
        # divide data to train_val/test set
        test_data = data_splits[k]
        if k==0:
            train_data = data_splits[0]
        else:
            train_data_splits=data_splits[0:k]+data_splits[k+1:k_fold]
            train_data = np.concatenate(train_data_splits)
        if compare_prune:
            # further split train_val into train/val set
            split_ratio = .8
            val_data = train_data[int(train_data.shape[0]*split_ratio):]
            train_data = train_data[:int(train_data.shape[0]*split_ratio)]
        # train model
        model.train(train_data[:, :-1], train_data[:, -1])
        cm_dict['unpruned'] = compute_confusion_matrix(model, test_data, len(set(test_data[:,-1])))
        if compare_prune:
            model.prune(val_data[:, :-1], val_data[:, -1])
            cm_dict['pruned'] = compute_confusion_matrix(model, test_data, len(set(test_data[:,-1])))
        yield model,cm_dict


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


def compute_measures(confusion_matrix,beta=1):
    """
    compute f_measure over a confusion matrix

    Args:
            confusion_matrix (np.ndarry): square 2-D matrix
            beta (int or float): weight of precision for computing f measures
    Outputs:
        measures (dict of :(np.array,lenghth=confusion_matrix's width)): {'precision':precision, 'recall':recall, 'f_measure':f_measure}
    """
    assert(type(confusion_matrix)==np.ndarray)
    assert(len(confusion_matrix.shape)==2)
    assert(confusion_matrix.shape[0]==confusion_matrix.shape[1])
    assert(type(beta)==float or type(beta)==int)

    acc = np.trace(confusion_matrix)/np.sum(confusion_matrix)

    true_positive_vec=np.diag(confusion_matrix)

    precision_vec=true_positive_vec/confusion_matrix.sum(axis=0)

    recall_vec=true_positive_vec/confusion_matrix.sum(axis=1)

    f_measure=(1+beta**2)*(precision_vec*recall_vec)/(beta**2*precision_vec+recall_vec)
    measures = {'precision':precision_vec, 'recall':recall_vec, 'f_measure':f_measure, 'acc':acc}
    return measures

def print_measures(measures):
    """print out the measures computed by compute_measures(confusion_matrix)"""
    assert(type(measures) == dict)
    assert('precision' in measures and 'recall' in measures and 'f_measure' in measures and 'acc' in measures)
    pr = measures['precision']
    print('Precision:')
    for i,p in enumerate(pr):
        print('        Class {}: {:.3f}'.format(i+1,p))
    re = measures['recall']
    print('Recall:')
    for i,r in enumerate(re):
        print('        Class {}: {:.3f}'.format(i+1,r))
    f_m = measures['f_measure']
    print('F Score:')
    for i,f in enumerate(f_m):
        print('        Class {}: {:.3f}'.format(i+1,f))
    print('The average accuracy = {:.3f}'.format(measures['acc']))

def main():
    # Load the datasets
    assert len(sys.argv) == 2, 'Usage: python train.py path/to/data/file'
    try:
        data = np.loadtxt(sys.argv[1])
    except:
        print('Please give the correct path to the dataset')
        exit()
    np.random.shuffle(data)

    tree = dt.Decision_tree()

    #Initialize confusion matrix
    un_total_cm = None
    pr_total_cm = None
    for _,cm_dict in cross_validate(tree, data, 5, compare_prune = True):
        un_cm = cm_dict['unpruned']
        un_total_cm = un_cm if un_total_cm is None else un_total_cm + un_cm
        pr_cm = cm_dict['pruned']
        pr_total_cm = pr_cm if pr_total_cm is None else pr_total_cm + pr_cm
    un_measures = compute_measures(un_total_cm)
    pr_measures = compute_measures(pr_total_cm)
    print('Performace on the dataset: '+sys.argv[1])
    print('-------------Unpruned Average Performance----------------------')
    print_measures(un_measures)
    print('--------------Pruned Average Performance-----------------------')
    print_measures(pr_measures)
    # Visualize the tree while training on the whole training set
    split_ratio = 0.8
    train_num = int(data.shape[0]*split_ratio)
    x_train = data[:train_num, :-1]
    y_train = data[:train_num, -1]

    x_test = data[train_num:, :-1]
    y_test = data[train_num:, -1]
    tree.train(x_train, y_train)

    tree.draw("Unpruned.png")

    tree.prune(x_test, y_test)

    tree.draw("Pruned.png")

if __name__=="__main__":
	main()



