import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve, auc


class HelperFunctions():

    def acheivement_score(rating):
        '''
        Takes the rating passed to it and returns an integer
        value representing how school meets target goals
        '''
        if rating == 'Exceeding Target':
            return 4
        if rating == 'Meeting Target':
            return 3
        if rating == 'Approaching Target':
            return 2
        if rating == 'Not Meeting Target':
            return 1
        else:
            return None

    def percent_cols_to_float(df):
        '''
        For any dataframe passed in, returns a new dataframe where
        values are floats between 0 and 1 representing the respective
        rate or percent in that column
        '''
        for col in df.columns:
            if 'Rate' in col or 'Percent' in col or '%' in col:
                df[col] = df[col].apply(
                    lambda x: float(x.replace('%', ''))*.01)
        return df

    def make_grades_int(grade):
        '''
        Takes a grade and returns an integer representative of that
        grade in the school system
        '''
        if grade == 'PK':
            return -1
        elif grade == '0K':
            return 0
        else:
            return int(grade)

    def grid_search_classifier(clf, param_grid, X_train, X_test, y_train, y_test, scoring='f1_weighted'):
        grid_clf = GridSearchCV(clf, param_grid, scoring=scoring)
        grid_clf.fit(X_train, y_train)

        best_parameters = grid_clf.best_params_

        print("Grid Search found the following optimal parameters: ")
        for param_name in sorted(best_parameters.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))

        y_pred = grid_clf.predict(X_test)

        print()
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print('Accuracy score:', round(accuracy_score(y_test, y_pred), 2))

        cm = confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],
                             index=['F', 'T'],
                             columns=['F', 'T'])
        plt.figure(figsize=(7, 5))
        sns.heatmap(df_cm, annot=True, cmap='Greens')
        plt.xlabel('Pred Val')
        plt.ylabel('True Val')
        plt.show()
        return grid_clf

    def plot_ROC(y_test, X_test, grid_clf):
        fpr, tpr, thresholds = roc_curve(
            y_test, grid_clf.predict_proba(X_test)[:, 1])

        print('AUC: {}'.format(auc(fpr, tpr)))
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.yticks([i/10.0 for i in range(11)])
        plt.xticks([i/10.0 for i in range(11)])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

    def drop_impractical_columns(df):
        cols_to_drop = ['Adjusted Grade',
                        'New?',
                        'Other Location Code in LCGMS',
                        'School Name',
                        'District',
                        'SED Code',
                        'Latitude',
                        'Longitude',
                        'Address (Full)',
                        'City',
                        'Zip',
                        'Grades',
                        'Rigorous Instruction Rating',
                        'Collaborative Teachers Rating',
                        'Supportive Environment Rating',
                        'Effective School Leadership Rating',
                        'Strong Family-Community Ties Rating',
                        'Trust Rating',
                        'School Income Estimate',
                        'Average ELA Proficiency',
                        'Community School?',
                        'Grade 3 ELA - All Students Tested',
                        'Grade 3 ELA 4s - All Students',
                        'Grade 3 ELA 4s - American Indian or Alaska Native',
                        'Grade 3 ELA 4s - Black or African American',
                        'Grade 3 ELA 4s - Hispanic or Latino',
                        'Grade 3 ELA 4s - Asian or Pacific Islander',
                        'Grade 3 ELA 4s - White',
                        'Grade 3 ELA 4s - Multiracial',
                        'Grade 3 ELA 4s - Limited English Proficient',
                        'Grade 3 ELA 4s - Economically Disadvantaged',
                        'Grade 3 Math - All Students tested',
                        'Grade 3 Math 4s - All Students',
                        'Grade 3 Math 4s - American Indian or Alaska Native',
                        'Grade 3 Math 4s - Black or African American',
                        'Grade 3 Math 4s - Hispanic or Latino',
                        'Grade 3 Math 4s - Asian or Pacific Islander',
                        'Grade 3 Math 4s - White',
                        'Grade 3 Math 4s - Multiracial',
                        'Grade 3 Math 4s - Limited English Proficient',
                        'Grade 3 Math 4s - Economically Disadvantaged',
                        'Grade 4 ELA - All Students Tested',
                        'Grade 4 ELA 4s - All Students',
                        'Grade 4 ELA 4s - American Indian or Alaska Native',
                        'Grade 4 ELA 4s - Black or African American',
                        'Grade 4 ELA 4s - Hispanic or Latino',
                        'Grade 4 ELA 4s - Asian or Pacific Islander',
                        'Grade 4 ELA 4s - White',
                        'Grade 4 ELA 4s - Multiracial',
                        'Grade 4 ELA 4s - Limited English Proficient',
                        'Grade 4 ELA 4s - Economically Disadvantaged',
                        'Grade 4 Math - All Students Tested',
                        'Grade 4 Math 4s - All Students',
                        'Grade 4 Math 4s - American Indian or Alaska Native',
                        'Grade 4 Math 4s - Black or African American',
                        'Grade 4 Math 4s - Hispanic or Latino',
                        'Grade 4 Math 4s - Asian or Pacific Islander',
                        'Grade 4 Math 4s - White',
                        'Grade 4 Math 4s - Multiracial',
                        'Grade 4 Math 4s - Limited English Proficient',
                        'Grade 4 Math 4s - Economically Disadvantaged',
                        'Grade 5 ELA - All Students Tested',
                        'Grade 5 ELA 4s - All Students',
                        'Grade 5 ELA 4s - American Indian or Alaska Native',
                        'Grade 5 ELA 4s - Black or African American',
                        'Grade 5 ELA 4s - Hispanic or Latino',
                        'Grade 5 ELA 4s - Asian or Pacific Islander',
                        'Grade 5 ELA 4s - White',
                        'Grade 5 ELA 4s - Multiracial',
                        'Grade 5 ELA 4s - Limited English Proficient',
                        'Grade 5 ELA 4s - Economically Disadvantaged',
                        'Grade 5 Math - All Students Tested',
                        'Grade 5 Math 4s - All Students',
                        'Grade 5 Math 4s - American Indian or Alaska Native',
                        'Grade 5 Math 4s - Black or African American',
                        'Grade 5 Math 4s - Hispanic or Latino',
                        'Grade 5 Math 4s - Asian or Pacific Islander',
                        'Grade 5 Math 4s - White',
                        'Grade 5 Math 4s - Multiracial',
                        'Grade 5 Math 4s - Limited English Proficient',
                        'Grade 5 Math 4s - Economically Disadvantaged',
                        'Grade 6 ELA - All Students Tested',
                        'Grade 6 ELA 4s - All Students',
                        'Grade 6 ELA 4s - American Indian or Alaska Native',
                        'Grade 6 ELA 4s - Black or African American',
                        'Grade 6 ELA 4s - Hispanic or Latino',
                        'Grade 6 ELA 4s - Asian or Pacific Islander',
                        'Grade 6 ELA 4s - White',
                        'Grade 6 ELA 4s - Multiracial',
                        'Grade 6 ELA 4s - Limited English Proficient',
                        'Grade 6 ELA 4s - Economically Disadvantaged',
                        'Grade 6 Math - All Students Tested',
                        'Grade 6 Math 4s - All Students',
                        'Grade 6 Math 4s - American Indian or Alaska Native',
                        'Grade 6 Math 4s - Black or African American',
                        'Grade 6 Math 4s - Hispanic or Latino',
                        'Grade 6 Math 4s - Asian or Pacific Islander',
                        'Grade 6 Math 4s - White',
                        'Grade 6 Math 4s - Multiracial',
                        'Grade 6 Math 4s - Limited English Proficient',
                        'Grade 6 Math 4s - Economically Disadvantaged',
                        'Grade 7 ELA - All Students Tested',
                        'Grade 7 ELA 4s - All Students',
                        'Grade 7 ELA 4s - American Indian or Alaska Native',
                        'Grade 7 ELA 4s - Black or African American',
                        'Grade 7 ELA 4s - Hispanic or Latino',
                        'Grade 7 ELA 4s - Asian or Pacific Islander',
                        'Grade 7 ELA 4s - White',
                        'Grade 7 ELA 4s - Multiracial',
                        'Grade 7 ELA 4s - Limited English Proficient',
                        'Grade 7 ELA 4s - Economically Disadvantaged',
                        'Grade 7 Math - All Students Tested',
                        'Grade 7 Math 4s - All Students',
                        'Grade 7 Math 4s - American Indian or Alaska Native',
                        'Grade 7 Math 4s - Black or African American',
                        'Grade 7 Math 4s - Hispanic or Latino',
                        'Grade 7 Math 4s - Asian or Pacific Islander',
                        'Grade 7 Math 4s - White',
                        'Grade 7 Math 4s - Multiracial',
                        'Grade 7 Math 4s - Limited English Proficient',
                        'Grade 7 Math 4s - Economically Disadvantaged',
                        'Grade 8 ELA - All Students Tested',
                        'Grade 8 ELA 4s - All Students',
                        'Grade 8 ELA 4s - American Indian or Alaska Native',
                        'Grade 8 ELA 4s - Black or African American',
                        'Grade 8 ELA 4s - Hispanic or Latino',
                        'Grade 8 ELA 4s - Asian or Pacific Islander',
                        'Grade 8 ELA 4s - White',
                        'Grade 8 ELA 4s - Multiracial',
                        'Grade 8 ELA 4s - Limited English Proficient',
                        'Grade 8 ELA 4s - Economically Disadvantaged',
                        'Grade 8 Math - All Students Tested',
                        'Grade 8 Math 4s - All Students',
                        'Grade 8 Math 4s - American Indian or Alaska Native',
                        'Grade 8 Math 4s - Black or African American',
                        'Grade 8 Math 4s - Hispanic or Latino',
                        'Grade 8 Math 4s - Asian or Pacific Islander',
                        'Grade 8 Math 4s - White',
                        'Grade 8 Math 4s - Multiracial',
                        'Grade 8 Math 4s - Limited English Proficient',
                        'Grade 8 Math 4s - Economically Disadvantaged'
                        ]
        return df.drop(cols_to_drop, axis=1)

    def rfe_test(classifier, features, X_train, y_train):
        ranking_list = []
        for i in range(50):
            clf = classifier
            rfecv = RFECV(clf).fit(X_train, y_train)
            ranking_list.append(rfecv.ranking_)
        return pd.DataFrame(zip(features, sum(ranking_list)/50)).sort_values(by=1)
