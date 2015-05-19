'''for creating models,We will deal with all the three coupons data.All the three coupons data(original data and extracted features) are loaded in different data frames and models are built for all three coupons independantly'''

%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from random import shuffle
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_curve,auc,classification_report,confusion_matrix,make_scorer,roc_auc_score,accuracy_score
from sklearn import pipeline

from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
import random
from random import shuffle
from sklearn.linear_model import LogisticRegression

FIGSIZE = (11, 7)

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 30)

# set some nicer defaults for matplotlib
from matplotlib import rcParams

#these colors come from colorbrewer2.org. Each is an RGB triplet
dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843),
                (0.4, 0.4, 0.4)]

rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 150
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.grid'] = False
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'none'


def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecesary plot borders and axis ticks
    
    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    
    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    
    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()
        
#DownSample the data to overcome class imbalance problem
def down_sample(x, y, ratio):
    pos_indices = [i for i in range(len(y)) if y[i] == 1]
    neg_indices = [i for i in range(len(y)) if y[i] == 0]

    neg_num = min(int(len(pos_indices) * ratio), len(neg_indices))
    shuffle(neg_indices)
    sample_indices = pos_indices + neg_indices[0 : neg_num]
    shuffle(sample_indices)

    # Down sampling.
    x_ds = [x[idx] for idx in sample_indices]
    y_ds = [y[idx] for idx in sample_indices]
    return (x_ds, y_ds)
    
#Load Coupon 1 Data Frame
train_df_coup1 = pd.read_csv('train_df_coup1.csv')
test_df_coup1 = pd.read_csv('test_df_coup1.csv')

#Plot ROC Curves
def plot_roc_curve(target_test,target_predicted_proba):
    fpr,tpr,thresholds = roc_curve(target_test,target_predicted_proba[:,1])
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label= 'roc_curve(area=%0.3f)' %(roc_auc))
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower_right")
    
#Gradient Boosted Classifier.Hyperparameters are found using Grid Search

train_df_coup1_y = train_df_coup1['coupon1Used']
train_coup1_X = train_df_coup1.drop('coupon1Used',1).values
train_coup1_y = train_df_coup1_y.values
test_coup1_X = test_df_coup1.values
std= StandardScaler()
train_coup1_X= std.fit_transform(train_coup1_X)
test_coup1_X= std.fit_transform(test_coup1_X)
X_train1,X_test1,y_train1,y_test1 = train_test_split(train_coup1_X,train_coup1_y,test_size = 0.20,random_state=42*5)
gbc1 = GradientBoostedClassifier(n_jobs=-1)
param_grid6 = {
              'n_estimators': [300,500,1000],
              'min_samples_split':[2,4,6],
              'max_features':[1,0.8],
              'min_samples_leaf':[1,3,5],
              'class_weight':[None,'auto','subsample']
             }
gs_cv1 = GridSearchCV(gbc1, param_grid6, cv=3,scoring='roc_auc',n_jobs=-1).fit(X_train6, y_train6)
print('Best hyperparameters: %r' % gs_cv1.best_params_)
gbc1.set_params(**gs_cv1.best_params_)
gbc1.fit(X_train6,y_train6)

#Building Models for Coupon 2

train_df_coup2 = pd.read_csv('train_df_coup2.csv')
test_df_coup2 = pd.read_csv('test_df_coup2.csv')

#Random Forest Model
train_df_coup2_y = train_df_coup2['coupon2Used']
train_coup2_X = train_df_coup2.drop('coupon2Used',1).values
train_coup2_y = train_df_coup2_y.values
test_coup2_X = test_df_coup2.values
std= StandardScaler()
train_coup2_X= std.fit_transform(train_coup2_X)
test_coup2_X= std.fit_transform(test_coup2_X)
X_train12,X_test12,y_train12,y_test12 = train_test_split(train_coup2_X,train_coup2_y,test_size = 0.20,random_state=42*12)
rf12 = RandomForestClassifier(max_features=1,min_samples_split=2,min_samples_leaf=5,n_estimators=500,n_jobs=-1)
param_grid12 = {
              'n_estimators': [300,500,1000],
              'min_samples_split':[2,4,6],
              'max_features':[1,0.8],
              'min_samples_leaf':[1,3,5],
              'class_weight':[None,'auto','subsample']
             }
gs_cv12 = GridSearchCV(rf12, param_grid12, cv=3,scoring='accuracy',n_jobs=-1).fit(X_train12, y_train12)
print('Best hyperparameters: %r' % gs_cv12.best_params_)
rf12.set_params(**gs_cv12.best_params_)
rf12.fit(X_train12,y_train12)
sorted(gs_cv12.grid_scores_,key=lambda x:x.mean_validation_score)

#Coupon 3

train_df_coup3 = pd.read_csv('train_df_coup3.csv')
test_df_coup3 = pd.read_csv('test_df_coup3.csv')

#Gradient Boosted Classifier

train_df_coup3_y = train_df_coup3['coupon3Used']
train_coup3_X = train_df_coup3.drop('coupon3Used',1).values
train_coup3_y = train_df_coup3_y.values
test_coup3_X = test_df_coup3.values
std= StandardScaler()
train_coup3_X= std.fit_transform(train_coup3_X)
test_coup3_X= std.fit_transform(test_coup3_X)
X_train14,X_test14,y_train14,y_test14 = train_test_split(train_coup3_X,train_coup3_y,test_size = 0.20)
gbc14 = GradientBoostingClassifier(n_estimators=3000)
param_grid14 = {'max_depth': [3,4,6],#tree depths
              'min_samples_leaf': [5,9,12], #no. of samples to be at leaf nodes
              'learning_rate': [0.1,0.01,0.05,0.001]## Shrinkage
              #'max_features': [1.0, 0.3] #no.of features before finding best split node #stochastic gradient 
              }
#loss
gs_cv14 = GridSearchCV(gbc14, param_grid14, cv=3,scoring='accuracy',n_jobs=-1).fit(X_train14, y_train14)
print('Best hyperparameters: %r' % gs_cv14.best_params_)
gbc14.set_params(**gs_cv14.best_params_)
gbc14.fit(X_train14,y_train14)

#Models Ensambling

def ensambling(X,Y):

    
    # The DEV SET will be used for all training and validation purposes
    # The TEST SET will never be used for training, it is the unseen set.
    dev_cutoff = len(Y) * 4/5
    X_dev = X[:dev_cutoff]
    Y_dev = Y[:dev_cutoff]
    X_test = X[dev_cutoff:]
    Y_test = Y[dev_cutoff:]
    
    # Our level 0 classifiers
    clfs = [
            GradientBoostingClassifier(n_estimators=5000,max_depth=4,min_samples_leaf=20,learning_rate=0.001),
            RandomForestClassifier(n_estimators=300,max_features=0.6,min_samples_split=4,min_samples_leaf=5),
            RandomForestClassifier(n_estimators=500,max_features=0.8,min_samples_split=6,min_samples_leaf=5,class_weight='auto'),
            SVC(probability=True,kernel='rbf',C=1000,gamma=0.001),
            SVC(kernel='rbf',C=1000,gamma=0.001,class_weight='auto'),
    ]
    
    # Ready for cross validation
    skf = list(StratifiedKFold(Y_dev, n_folds))
    
    # Pre-allocate the data
    blend_train = np.zeros((X_dev.shape[0], len(clfs))) # Number of training data x Number of classifiers
    blend_test = np.zeros((X_test.shape[0], len(clfs))) # Number of testing data x Number of classifiers
    
    print 'X_test.shape = %s' % (str(X_test.shape))
    print 'blend_train.shape = %s' % (str(blend_train.shape))
    print 'blend_test.shape = %s' % (str(blend_test.shape))
    
    # For each classifier, we train the number of fold times (=len(skf))
    for j, clf in enumerate(clfs):
        print 'Training classifier [%s]' % (j)
        blend_test_j = np.zeros((X_test.shape[0], len(skf))) # Number of testing data x Number of folds , we will take the mean of the predictions later
        for i, (train_index, cv_index) in enumerate(skf):
            print 'Fold [%s]' % (i)
            
            # This is the training and validation set
            X_train = X_dev[train_index]
            Y_train = Y_dev[train_index]
            X_cv = X_dev[cv_index]
            Y_cv = Y_dev[cv_index]
            
            clf.fit(X_train, Y_train)
            print "cv_score= %d" % (clf.score(X_cv,Y_cv))
            # This output will be the basis for our blended classifier to train against,
            # which is also the output of our classifiers
            blend_train[cv_index, j] = clf.predict(X_cv)
            blend_test_j[:, i] = clf.predict(X_test)
            
        # Take the mean of the predictions of the cross validation set
        return blend_test_j
        blend_test[:, j] = blend_test_j.mean(1)
    
    print 'Y_dev.shape = %s' % (Y_dev.shape)
    
    
    # Start blending!
    bclf = LogisticRegression()
    bclf.fit(blend_train, Y_dev)
    
    # Predict now
    Y_test_predict = bclf.predict(blend_test)
    score = accuracy_score(Y_test, Y_test_predict)
    print 'Accuracy = %s' % (score)
    
    return score
    
    best_score = 0.0
    
for i in xrange(1):
    print 'Iteration [%s]' % (i)
    #random.shuffle(X1,X2)
    score = run_ensemble(X,Y)
    best_score = max(best_score, score)
    print
        
print 'Best score = %s' % (best_score)

#The input to ensamble function is the data and the assosciated labels
