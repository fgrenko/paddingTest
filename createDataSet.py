import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42

### Returns lists of DataSet objects (for X and y data) - containing their respective fold index, train, test and validation data sets
def createDataSets():
    # 1: Dataset preparation
    df = pd.read_csv('amp.csv')
    sequences = df['sequence'].tolist() # X
    labels = df['label'].tolist() # y

    amino_acids = list(set(''.join(sequences)))
    amino_to_num = {amino: num for num, amino in enumerate(amino_acids)}
    numerical_sequences = [[amino_to_num[amino] for amino in sequence] for sequence in sequences]

    # Stratified K fold - 10 fold
    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True) # training vs test -> 90% - 10%
    skf.get_n_splits(numerical_sequences, labels)

    # Split data into 10 folds of training and test data
    skf_split = skf.split(numerical_sequences, labels)

    XdataSets = []
    ydataSets = []

    for i, (train_indices, test_indices) in enumerate(skf_split):
        xds = DataSet(fold=i, testData=[numerical_sequences[i] for i in test_indices])
        yds = DataSet(fold=i, testData=[labels[i] for i in test_indices])

        x = [numerical_sequences[i] for i in train_indices]
        y = [labels[i] for i in train_indices]

        X_train, X_val, y_train, y_val = train_test_split(x, y, stratify=y, test_size=0.11, random_state=42)

        xds.setTrainingDataSet(X_train)
        xds.setValidationDataSet(X_val)
        yds.setTrainingDataSet(y_train)
        yds.setValidationDataSet(y_val)

        XdataSets.append(xds)
        ydataSets.append(yds)


    return XdataSets, ydataSets

class DataSet:
    def __init__(self, fold, testData, trainData=None, valData=None):
        self.fold = fold
        self.testData = testData
        self.trainData = trainData
        self.valData = valData

    def setValidationDataSet(self, valDataSet):
        self.valData = valDataSet

    def setTrainingDataSet(self, trainDataSet):
        self.trainData = trainDataSet

    def getTestDataSet(self):
        return self.testData

    def getValidationDataSet(self):
        return self.valData

    def getTrainingDataSet(self):
        return self.trainData