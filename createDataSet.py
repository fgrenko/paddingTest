import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42

### Returns lists of DataSet objects (for X and y data) - containing their respective fold index, train, test and validation data sets
### TODO: remove reading data in current function - input file path and table keys (String, String, String)
def createDataSets():
    # 1: Priprema skupa podataka
    df = pd.read_csv('amp.csv')
    sequences = df['sequence'].tolist() # X
    labels = df['label'].tolist() # y

    amino_acids = list(set(''.join(sequences)))
    # Staviti da se enumerira od 1
    amino_to_num = {amino: num for num, amino in enumerate(amino_acids)}
    numerical_sequences = [[amino_to_num[amino] for amino in sequence] for sequence in sequences]

    # print("Len of num_seq:")
    # print(len(numerical_sequences))

    # Stratified K fold - 10 fold
    # Remove -(set to None)- random state after testing, random_state=1
    # ValueError: Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True.
    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True) # training vs test -> 90% - 10%
    skf.get_n_splits(numerical_sequences, labels)

    # Split data into 10 folds of trainig and test data
    skf_split = skf.split(numerical_sequences, labels)

    # print(skf_split)

    # Format: i - index folda, train_index - index num_seq, test_index - index num_seq
    ## Need to: split train_index list to train and val

    XdataSets = []
    ydataSets = []

    for i, (train_indices, test_indices) in enumerate(skf_split):
        # [test_list[i] for i in index_list]
        xds = DataSet(fold=i, testData=[numerical_sequences[i] for i in test_indices])
        yds = DataSet(fold=i, testData=[labels[i] for i in test_indices])
        # xds = DataSet(fold=i, testData=np.take(numerical_sequences, test_indices))
        # yds = DataSet(fold=i, testData=np.take(labels, test_indices))

        x = [numerical_sequences[i] for i in train_indices]
        y = [labels[i] for i in train_indices]

        # print("Postotak trainig seta: ", len(x) / len(numerical_sequences))

        # TODO: get actual percentage
        # 10000 -> sfk: 9000:1000
        # 9000 -> traintest: 8100:900

        # test_size == val_size
        # tweak test_size to conform to initial skf 90-10 split
        X_train, X_val, y_train, y_val = train_test_split(x, y, stratify=y, test_size=0.11, random_state=42)

        xds.setTrainingDataSet(X_train)
        xds.setValidationDataSet(X_val)
        yds.setTrainingDataSet(y_train)
        yds.setValidationDataSet(y_val)

        XdataSets.append(xds)
        ydataSets.append(yds)

        # print("X set for fold: ", i)
        # print(len(xds.getTrainingDataSet()))
        # print(len(xds.getValidationDataSet()))
        # print(len(xds.getTestDataSet()))


    # print(len(XdataSets))
    # print(len(ydataSets))

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
