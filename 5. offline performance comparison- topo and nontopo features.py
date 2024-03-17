import numpy as np

from datetime import datetime
import csv
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

#%% offline performance comparison for imbalanced datasets and ML algorithms
ablation_settings = ['None'] #'All','Title','Department','Manager','None'
batch_size= 1024
hidden_channels=[256] #16,32,64,128,256
Epoch=1000
imbalance_settings = ['55','37','19','original'] # '55','37','19','original'

feature_settings = ['GraphEmbedding'] # 'GraphEmbedding', 'OriginalFeature','Both'

for hidden_channel in hidden_channels:
    for ablation_setting in ablation_settings:
        setting = ablation_setting + '_BS_'+ str(batch_size) + '_HC_' + str(hidden_channel)
        # % offline learning: classifiers comparison
        result_dir = "./data/results/" + ablation_setting + '_HC_' + str(hidden_channel) + "_offline_classification_results.csv"
        # result_dir = "./data/results/" + ablation_setting + "_offline_classification_results.csv"
        with open(result_dir, 'a', newline='') as f:
            writer = csv.writer(f)
            re_list = ['Classifier', 'ablation_setting', 'imbalance_setting', 'feature_setting','hidden_channel','Duration',
                       'test_cm', 'test_acc', 'test_pre', 'test_rec', 'test_f1',
                       'test_class1_pre', 'test_class1_rec', 'test_class1_f1',
                       'test_class0_pre', 'test_class0_rec', 'test_class0_f1',
                       'train_cm', 'train_acc', 'train_pre', 'train_rec', 'train_f1',
                       'train_class1_pre', 'train_class1_rec', 'train_class1_f1',
                       'train_class0_pre', 'train_class0_rec', 'train_class0_f1']
            writer.writerow(re_list)

        for imbalance_setting in imbalance_settings:

            './data/dataset/' + ablation_setting + '_' + imbalance_setting + '_HC_' + str(
                hidden_channel) + '_dataset_dict.pickle'
            # if hidden_channel == 64:
            #     with open('./data/dataset/' + ablation_setting + '_' + imbalance_setting + '_dataset_dict.pickle', 'rb') as f:
            #         dataset_dict = pickle.load(f)
            # else:
            with open('./data/dataset/' + ablation_setting + '_' + imbalance_setting + '_HC_' + str(hidden_channel) + '_dataset_dict.pickle', 'rb') as f:
                    dataset_dict = pickle.load(f)

            y=dataset_dict['y']
            y = y.astype(np.int32)
            x_User_embeddings = dataset_dict['x_User_embeddings']
            x_User_feat = dataset_dict['x_User_feat']
            x_Resource_embeddings = dataset_dict['x_Resource_embeddings']
            x_Resource_feat = dataset_dict['x_Resource_feat']


            for feature_setting in feature_settings:
                if feature_setting == 'GraphEmbedding':
                    X = np.concatenate((x_User_embeddings, x_Resource_embeddings), axis=1)
                elif feature_setting == 'OriginalFeature':
                    X = np.concatenate((x_User_feat, x_Resource_feat), axis=1)
                elif feature_setting == 'Both':
                    x_1 = np.concatenate((x_User_embeddings, x_Resource_embeddings), axis=1)
                    x_2 = np.concatenate((x_User_feat, x_Resource_feat), axis=1)
                    X = np.concatenate((x_1, x_2), axis=1)

                test_size = 0.33
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                print('test_size=', test_size)

                c_LR = LogisticRegression(random_state=0, solver='liblinear', max_iter=1000)
                c_RF = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=0)
                c_SVM = svm.LinearSVC(random_state=0, max_iter=1000)
                c_NN = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(64), activation='logistic', random_state=0,
                                     learning_rate='adaptive', max_iter=1000, shuffle=False, early_stopping=True)
                c_GNB = GaussianNB()
                n_features = X.shape[1]
                n_classes = 2

                classifier_list = [c_RF]  # c_LR, c_RF, c_SVM, c_NN, c_GNB
                classifier_name_list = ["c_RF"]  #"c_LR", "c_RF", "c_SVM", "c_NN", "c_GNB"

                for c in range(len(classifier_list)):
                    classifier = classifier_list[c]
                    classifier_name = classifier_name_list[c]

                    # classifier train
                    start_time = datetime.now()
                    classifier.fit(X_train, y_train)
                    predictions = classifier.predict(X_test)
                    # evaluate classifier on the test set
                    test_confusion_matrix = confusion_matrix(y_test, predictions)
                    report = classification_report(y_test, predictions, labels=[0, 1], target_names=['class 0', 'class 1'],
                                                   output_dict=True, zero_division=0)  # output_dict=True
                    test_acc = report['accuracy']
                    test_pre = report['macro avg']['precision']
                    test_rec = report['macro avg']['recall']
                    test_f1 = report['macro avg']['f1-score']
                    test_class1_pre = report['class 1']['precision']
                    test_class1_rec = report['class 1']['recall']
                    test_class1_f1 = report['class 1']['f1-score']
                    test_class0_pre = report['class 0']['precision']
                    test_class0_rec = report['class 0']['recall']
                    test_class0_f1 = report['class 0']['f1-score']
                    # % evaluate on the train set
                    predictions = classifier.predict(X_train)
                    train_confusion_matrix = confusion_matrix(y_train, predictions)
                    report = classification_report(y_train, predictions, labels=[0, 1], target_names=['class 0', 'class 1'],
                                                   output_dict=True, zero_division=0)  # output_dict=True
                    train_acc = report['accuracy']

                    train_pre = report['macro avg']['precision']
                    train_rec = report['macro avg']['recall']
                    train_f1 = report['macro avg']['f1-score']
                    train_class1_pre = report['class 1']['precision']
                    train_class1_rec = report['class 1']['recall']
                    train_class1_f1 = report['class 1']['f1-score']
                    train_class0_pre = report['class 0']['precision']
                    train_class0_rec = report['class 0']['recall']
                    train_class0_f1 = report['class 0']['f1-score']
                    end_time = datetime.now()
                    Duration=end_time - start_time

                    # save the results
                    with open(result_dir, 'a', newline='') as f:
                        writer = csv.writer(f)
                        re_list = [classifier_name, ablation_setting,imbalance_setting,feature_setting, hidden_channel,Duration,test_confusion_matrix, test_acc, test_pre, test_rec, test_f1,
                                   test_class1_pre, test_class1_rec, test_class1_f1, test_class0_pre, test_class0_rec,
                                   test_class0_f1,
                                   train_confusion_matrix, train_acc, train_pre, train_rec, train_f1, train_class1_pre,
                                   train_class1_rec, train_class1_f1, train_class0_pre, train_class0_rec, train_class0_f1]
                        writer.writerow(re_list)

                    print([classifier_name, ablation_setting,imbalance_setting,feature_setting])
                    print('test set performance:',
                          [test_confusion_matrix, test_acc, test_pre, test_rec, test_f1, test_class1_pre,
                           test_class1_rec, test_class1_f1, test_class0_pre, test_class0_rec,
                           test_class0_f1])

                    print('train set performance:', [train_confusion_matrix, train_acc, train_pre, train_rec, train_f1,
                                                     train_class1_pre, train_class1_rec, train_class1_f1, train_class0_pre,
                                                     train_class0_rec, train_class0_f1])

                    print('Duration: {}'.format(end_time - start_time))


