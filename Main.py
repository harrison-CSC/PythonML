# This is a machine learning project that compares the accuracy of random forest models with various configurations
# This is based on a dataset used in a machine learning course at SFSU
# Data originates from https://tadpole.grand-challenge.org/Data/

import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics as skmetrics
from plotnine import ggplot, aes, geom_tile, geom_text, ggtitle
from xgboost import XGBClassifier


url = "./DataSet.csv"
data = pandas.read_csv(url)

print('Here are the first 5 rows of data')
print(data.head(5))
print()

labels = data["DX"] # 'DX' refers to patient diagnosis
features = data.drop(columns=['PTID','DX']) # Patient ID and diagnosis are removes from the features data

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 1) # 20% of data used for testing



accuracylist = [] # Will store the % accuracy for each iteration
num_features_list = []
max_depthlist = []

currentBestRF = {'accuracy': 0, 'max_depth': 0, 'num_feat': 0}

iterator = 1

for md in [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50]:
    for num_features in range(1, 12):
        rf = RandomForestClassifier(n_estimators = 100, max_features = num_features, max_depth = md, bootstrap = True, random_state = 1)
        rf.fit(features_train, labels_train)

        labels_pred = rf.predict(features_test)
        accuracy = round(skmetrics.accuracy_score(labels_test, labels_pred), 2)

        if accuracy > currentBestRF['accuracy']:
            currentBestRF['accuracy'] = accuracy
            currentBestRF['max_depth'] = md
            currentBestRF['num_feat'] = num_features

        accuracylist.append(accuracy)
        num_features_list.append(num_features)
        max_depthlist.append(md)
    print('random forest: %.1f%% done' % (100 * iterator / 11))
    iterator += 1

data = {'Num_features': num_features_list, 'MaxDepth': max_depthlist, 'Accuracy': accuracylist}
df_features_depth = pandas.DataFrame(data)

plot1 = (ggplot(df_features_depth, aes('factor(Num_features)', 'factor(MaxDepth)', fill = 'Accuracy'))
 + ggtitle('RandomForestClassifier heatmap')
 + geom_tile(aes(width = 0.95, height = 0.95))
 + geom_text(aes(label = 'Accuracy'), size = 10)
)



accuracylist = []
learningrate_list = []
max_depthlist = []

iterator = 1

currentBestXGB = {'accuracy': 0, 'max_depth': 0, 'learning_rate': 0}

for md in [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50]:
    lr = 0.01
    while lr <= 0.2:
        xgb = XGBClassifier(max_depth = md, learning_rate = lr)

        # Note: errors with the below line may be caused by having a too recent version of xgboost. Try version 0.90 if errors occur
        xgb.fit(features_train, labels_train)

        labels_pred = xgb.predict(features_test)
        accuracy = round(skmetrics.accuracy_score(labels_test, labels_pred), 2)

        if accuracy > currentBestXGB['accuracy']:
            currentBestXGB['accuracy'] = accuracy
            currentBestXGB['max_depth'] = md
            currentBestXGB['learning_rate'] = lr

        accuracylist.append(accuracy)
        learningrate_list.append(lr)
        max_depthlist.append(md)
        lr = round((lr + 0.02), 2)
    print('xgb: %.1f%% done' % (100 * iterator / 11))
    iterator += 1

data = {'Learning_rate': learningrate_list, 'MaxDepth': max_depthlist, 'Accuracy': accuracylist}
df_features_depth = pandas.DataFrame(data)

plot2 = (ggplot(df_features_depth, aes('factor(Learning_rate)', 'factor(MaxDepth)', fill = 'Accuracy'))
 + ggtitle('XGBClassifier heatmap')
 + geom_tile(aes(width = 0.95, height = 0.95))
 + geom_text(aes(label = 'Accuracy'), size = 10)
)

print()
print('The highest accuracy achieved with the random forest classifier was', currentBestRF['accuracy'])
print('This was at max_depth:', currentBestRF['max_depth'], 'and num_features:', currentBestRF['num_feat'])

print()
print('The highest accuracy achieved with the extreme gradient booster was', currentBestXGB['accuracy'])
print('This was at max_depth:', currentBestXGB['max_depth'], 'and learning_rate:', currentBestXGB['learning_rate'])

print(plot1)
print(plot2)