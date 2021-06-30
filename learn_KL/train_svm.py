import pandas as pd
from matplotlib import cm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, labels, pos, fmt, normalisation=False, title="Confusion Matrix"):
    
    if normalisation:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    ax= plt.subplot(pos)
    sns.heatmap(cm, annot=True, ax = ax, fmt=fmt, cmap='Greens') #annot=True to annotate cells

    # labels, title and ticks
    ax.set_ylim([0,2])
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels') 
    ax.set_title(title) 
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)


def evaluate_performance(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision =  tp / (tp + fp)
    recall = tp / (tp + fn)
    print("Overall stats:")
    TPR, TNR = tp / (tp + fn), tn / (tn + fp)
    FPR, FNR = 1 - TNR, 1 - TPR
    print("TPR: %.3f, TNR: %.3f, \nFPR: %.3f, FNR: %.3f" % (TPR, TNR, FPR, FNR))
    print("Precision: %.3f" % precision)
    print("Recall: %.3f" % recall)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred))
    
    plt.figure(figsize=(15,5))
    plot_confusion_matrix(cm, labels, '120', 'g')
    plot_confusion_matrix(cm, labels, '121', '.2f', normalisation=True, title="Normalised Confusion Matrix")


def plot_contours(ax, clf, xx, yy, levels, **params):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, **params)
    cs = plt.contour(xx, yy, Z, levels=levels, linewidths=1, 
                linestyles='dashed', origin='lower')
    ax.clabel(cs, fmt='%2.2f', colors='k', fontsize=14)

def make_meshgrid(x, y, h=.1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_svm_decision_boundary(X_pca, y, clf, levels, title):
    """
    X_pca : either X_train_pca to view the model or X_test_pca to view predictions
    y : either y_train to view the original training targets, y_test to view original test targets,
        y_pred to view predictions, y_pred_adj to view predictions after adjustment
    """
    X0, X1 = X_pca[:, 0], X_pca[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    # plot figure
    fig, ax = plt.subplots(figsize=(12,9))
    # fig.patch.set_facecolor('white')
    cdict1={0:'blue',1:'orange'}

    Y_tar_list = y.tolist()
    yl1= [int(target1) for target1 in Y_tar_list]
    labels1=yl1

    labl1={0:'Incorrect edge pairs',1:'Correct edge pairs'}
    marker1={0:'+',1:'d'}
    alpha1={0:.8, 1:0.5}

    levels=levels

    plot_contours(ax, clf, xx, yy, levels, cmap='coolwarm', alpha=0.7)

    for l1 in np.unique(labels1):
        ix1=np.where(labels1==l1)
        ax.scatter(X0[ix1],X1[ix1], c=cdict1[l1],label=labl1[l1],s=70,marker=marker1[l1],alpha=alpha1[l1])

    plt.legend(fontsize=15)
    plt.xlabel("1st Principal Component",fontsize=14)
    plt.ylabel("2nd Principal Component",fontsize=14)
    plt.title(title)
    plt.show()


def downsample(df, max_size):
    seed = int(np.random.uniform() * 100)
    if len(df) > max_size:
        frac = max_size / (len(df))
        df = df.sample(frac=frac, replace=True, random_state=seed)
    return df


# load data
filename = "output/track_sim/sigma0.5/10000_events_training_data.csv"
df = pd.read_csv(filename)

df = downsample(df, 1000)
correct_pairs = df.loc[df['truth'] == 1]
incorrect_pairs = df.loc[df['truth'] == 0]
incorrect_pairs = downsample(incorrect_pairs, len(correct_pairs))

Xc = correct_pairs[['kl_dist', 'emp_var']]
yc = correct_pairs['truth']
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.3, random_state=42)

Xi = incorrect_pairs[['kl_dist', 'emp_var']]
yi = incorrect_pairs['truth']
Xi_train, Xi_test, yi_train, yi_test = train_test_split(Xi, yi, test_size=0.3, random_state=42)

X_train = pd.concat([Xc_train, Xi_train], ignore_index=False)
y_train = pd.concat([yc_train, yi_train], ignore_index=False)
X_test = pd.concat([Xc_test, Xi_test], ignore_index=False)
y_test = pd.concat([yc_test, yi_test], ignore_index=False)

# grid search cv to find optimum hyperparameters
pipe_steps = [('SVM', SVC(class_weight='balanced'))]
# check_params = {
#     'SVM__kernel' : ['linear', 'rbf', 'poly'],
#     'SVM__degree' : [2, 3],
#     'SVM__C': [0.01, 0.1, 1.0, 10.0],
#     'SVM__gamma': [0.01, 0.1, 1.0, 10.0]
# }
check_params = {
    'SVM__kernel' : ['poly'],
    'SVM__degree' : [2],
    'SVM__C': [1.0],
    'SVM__gamma': [1.0]
}

pipeline = Pipeline(pipe_steps)
print("Start fitting training data...")

create_grid = GridSearchCV(pipeline, param_grid=check_params, cv=2, 
                           scoring='recall', verbose=1, n_jobs=-1)
create_grid.fit(X_train, y_train)

# apply the trained svm to the test data
print("Score for 4 fold CV := %3.2f" %(create_grid.score(X_test, y_test)))
print("Best fit parameters from training data:")
print(create_grid.best_params_)
print("Cross-val complete")
print(create_grid.best_estimator_)
print(create_grid.best_score_)


y_pred = create_grid.predict(X_test)
evaluate_performance(y_test, y_pred, ['0', '1'])

C = create_grid.best_params_['SVM__C']
gamma = create_grid.best_params_['SVM__gamma']
kernel = create_grid.best_params_['SVM__kernel']
degree = create_grid.best_params_['SVM__degree']

clf = SVC(kernel=kernel, class_weight='balanced', C=C, gamma=gamma)
model = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
title = "SVM Decision Boundary Predictions, "+str(kernel)+" degree "+str(degree)+" kernel, C="+str(C)+", gamma="+str(gamma)
plot_svm_decision_boundary(X_test, y_pred, clf, levels, title)