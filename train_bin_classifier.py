# train_classifier.py - Template train binary classifier.

import pandas


def create_baseline():
    # create model
    model = Sequential()
    model.add(Flatten(input_shape=(1,)))
    model.add(Dense(60, input_dim=1, activation='relu'))
    # model.add(Dense(60, input_dim=60, activation='relu'))
    # model.add(Dense(30, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def load_sonar_dataset ():
    # load dataset
    dataframe = pandas.read_csv("sonar.csv", header=None)
    dataset = dataframe.values

    # split into input (X) and output (Y) variables
    X = dataset[:,0:60].astype(float)
    Y = dataset[:,60]
    return (X, Y)


def load_test_needs_mark_dataset ():
    # load dataset
    dataframe = pandas.read_csv("test_needs_marks.csv", header=None)
    dataset = dataframe.values

    # split into input (X) and output (Y) variables
    X = dataset[:,0:1].astype(str)
    Y = dataset[:,1]

    print(X)
    return (X, Y)


if __name__ == "__main__":
    import sys
    # X, Y = load_sonar_dataset()
    X, Y = load_test_needs_mark_dataset()

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Flatten
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline


    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    print(encoded_Y)

    # evaluate model with standardized dataset
    estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))