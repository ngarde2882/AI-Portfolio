import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

'''
Problem: University Admission Classification using SVMs

Instructions:
1. Do not use any additional libraries. Your code will be tested in a pre-built environment with only 
   the library specified in question instruction available. Importing additional libraries will result in 
   compilation errors and you will lose marks.

2. Fill in the skeleton code precisely as provided. You may define additional 
   default arguments or helper functions if necessary, but ensure the input/output format matches.

3. Save your best model as 'svm_best_model.pkl' as shown in the main function.
'''

def create_binary_label(df: pd.DataFrame, column: str) -> pd.DataFrame:
    '''
    Convert the values in the specified column to binary labels:
    - Values greater than the median will be labeled as 1.
    - Values less than or equal to the median will be labeled as 0.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name based on which the binary label is created.

    Returns:
        pd.DataFrame: DataFrame with an additional binary label column.
    '''
    med = df[column].median()
    label = pd.Series(df['Chance of Admit '].map(lambda x:1 if x>med else 0),name='binary label')
    return pd.concat([df,label],axis=1)
    

def split_data(df: pd.DataFrame, features: list, label: str) -> tuple:
    '''
    Split the data into training and testing sets.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        features (list): List of column names to use as features.
        label (str): The column name for the label.
        test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
        random_state (int): Seed for reproducibility (default is 42).

    Returns:
        tuple: X_train, X_test, y_train, y_test (numpy arrays)
    '''
    X,Xt,y,yt = train_test_split(df[features], label, test_size=0.2, random_state=42)
    return (X,Xt,y,yt)

def train_svm_model(X_train: np.ndarray, y_train: np.ndarray, kernel: str = 'linear') -> SVC:
    '''
    Train an SVM model using the specified kernel.

    Parameters:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        kernel (str): The kernel type to be used in the algorithm (default is 'linear').
    '''
    svc = SVC(C=5, kernel=kernel)
    svc.fit(X_train,y_train)
    return svc


if __name__ == "__main__":

    data_train = pd.read_csv('C:/Users/nick2/Desktop/School Stuff/Fall24/633 Machine Learning/hws/hw3/data_train-2.csv', index_col=0)
    data_train.dropna(inplace=True)
    data_train = data_train.astype('float64')
    data_train = create_binary_label(data_train, 'Chance of Admit ')
    X = data_train.loc[:,data_train.columns.difference(['Chance of Admit ','binary label'])]
    y = data_train.loc[:,['binary label']]
    X, Xt, y, yt = split_data(X,X.columns,y)
    # print(X.shape,y.shape)

    # from sklearn.model_selection import train_test_split, GridSearchCV
    # from sklearn.metrics import accuracy_score
    # features = [['CGPA','LOR ','GRE Score'],
    #             ['CGPA','GRE Score','SOP','LOR '],
    #             ['CGPA','GRE Score','SOP','LOR ','University Rating'],
    #             ['CGPA','GRE Score','SOP','LOR ','Research'],
    #             ['CGPA','GRE Score','SOP','LOR ','TOEFL Score'],
    #             ['CGPA','GRE Score','SOP','LOR ','Research','University Rating'],
    #             ['CGPA','GRE Score','SOP','Research','University Rating'],
    #             ['CGPA','GRE Score','LOR ','Research','University Rating'],
    #             ['GRE Score','LOR ','Research','University Rating'],
    #             ['GRE Score','LOR ','Research','University Rating','TOEFL Score']]
    # D = dict()
    # for feat in features:
    #     print(feat)
    #     key = ','.join(feat)
    #     tuned_parameters = [{'C': [1, 3, 5, 7, 10, 100], 'kernel':['linear','rbf','poly']}]
    #     clf = GridSearchCV(SVC(degree=3), tuned_parameters, scoring='accuracy')
    #     clf.fit(np.array(X[feat]),np.array(y).ravel())
    #     D[key] = {'params':clf.best_params_,'acc':float(clf.best_score_)}
    #     print(clf.best_params_)
    #     pred = SVC(C=clf.best_params_['C'], kernel='linear').fit(np.array(X[feat]),np.array(y).ravel()).predict(np.array(Xt[feat]))
    #     D[key]['tacc'] = round(accuracy_score(yt,pred),3)
    
    # print('Display:')
    # for i in D.keys():
    #     print(i)
    #     print(f' {D[i]['params']}')
    #     print(f' {D[i]['acc']}')
    #     print(f' {D[i]['tacc']}')
#     CGPA,LOR ,GRE Score
#  {'C': 5, 'kernel': 'linear'}
#  0.8867269984917044
#  0.781
# CGPA,GRE Score,SOP,LOR
#  {'C': 7, 'kernel': 'linear'}
#  0.8984917043740573
#  0.797
# CGPA,GRE Score,SOP,LOR ,University Rating
#  {'C': 1, 'kernel': 'linear'}
#  0.9024132730015083
#  0.812
# CGPA,GRE Score,SOP,LOR ,Research
#  {'C': 10, 'kernel': 'linear'}
#  0.9062594268476621
#  0.812
# CGPA,GRE Score,SOP,LOR ,TOEFL Score
#  {'C': 3, 'kernel': 'linear'}
#  0.8984917043740573
#  0.781
# CGPA,GRE Score,SOP,LOR ,Research,University Rating
#  {'C': 10, 'kernel': 'linear'}
#  0.9101809954751131
#  0.812
# CGPA,GRE Score,SOP,Research,University Rating
#  {'C': 100, 'kernel': 'linear'}
#  0.8750377073906487
#  0.828
# CGPA,GRE Score,LOR ,Research,University Rating
#  {'C': 1, 'kernel': 'linear'}
#  0.9024132730015083
#  0.828
# GRE Score,LOR ,Research,University Rating
#  {'C': 1, 'kernel': 'linear'}
#  0.8828054298642535
#  0.844
# GRE Score,LOR ,Research,University Rating,TOEFL Score
#  {'C': 5, 'kernel': 'linear'}
#  0.8829562594268477
#  0.797

    # CGPA and GRE Score were a good base from the homework
    # Research and University Rating work well and better together
    # TOEFL doesnt contribute with the whole
    # SOP and LOR work best together

    # For each SVM kernel, train a model with possible feature combinations and store the best model in the best_model variable
    best_model = SVC(C=1, kernel='linear').fit(np.array(X[['GRE Score','LOR ','Research','University Rating']]),np.array(y).ravel())

    # Save the best model to the pickle file
    with open('svm_best_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    # Attach the pkl file with the submission