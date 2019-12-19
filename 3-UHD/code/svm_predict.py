import numpy as np
import os
from sklearn import svm


for seed in range(10):
    np.random.seed(seed)

    path = 'images_feature/combined/10_dim'

    train_example_num = 100

    file_name = os.path.join(path, "samples.txt")
    temp_arr = np.loadtxt(file_name)
    np.random.shuffle(temp_arr)
    np.savetxt(os.path.join(path, "shuffled_train.txt"), temp_arr[:train_example_num])
    np.savetxt(os.path.join(path, "shuffled_test.txt"), temp_arr[train_example_num:])

    train_arr = np.loadtxt(os.path.join(path, "shuffled_train.txt"))
    test_arr = np.loadtxt(os.path.join(path, "shuffled_test.txt"))

    clf = svm.SVC()

    train_x = train_arr[:, :-1]
    train_y = train_arr[:, -1]
    clf.fit(train_x, train_y)

    print("positive examples: {0}, negative examples: {1}".format(list(train_y).count(1), list(train_y).count(0)))

    test_x = test_arr[:, :-1]
    test_y = test_arr[:, -1]
    predicted_y = clf.predict(test_x)

    sample_num = len(predicted_y)

    combined_list = [(test_y[i], predicted_y[i]) for i in range(sample_num)]
    tp = combined_list.count((1, 1))
    tn = combined_list.count((0, 0))
    fp = combined_list.count((0, 1))
    fn = combined_list.count((1, 0))
    print("true positive = {0}, true negative = {1}, false positive = {2}, false negative = {3}".format(
        tp, tn, fp, fn))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("precision = {0}, recall = {1}".format(precision, recall))
    f1 = 2 * precision * recall / (precision + recall)
    print("f1 score = ", f1)
