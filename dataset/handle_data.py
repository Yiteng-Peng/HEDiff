import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.datasets import load_digits
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from sklearn.utils import shuffle


TEST_SIZE = 0.2


def load_bank(data_path):
    data = pd.read_csv(data_path, sep=";")
    data.replace("unknown", np.nan, inplace=True)
    data.dropna(inplace=True)
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]
    categorical_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month",
                        "day_of_week", "poutcome"]
    for col in categorical_cols:
        label_encoder = LabelEncoder()
        features[col] = label_encoder.fit_transform(features[col])
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    features = features.values.astype(np.float32)
    labels = labels.astype(np.int32)

    return features, labels


def load_credit(data_path):
    data = pd.read_csv(data_path)
    features = data.iloc[:, 1:-1]
    labels = data.iloc[:, -1]

    features = features.values.astype(np.float32)
    labels = labels.astype(np.int32)

    return features, labels


def load_car(data_path):
    data = pd.read_csv(data_path, header=None)
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]

    for col in range(6):
        label_encoder = LabelEncoder()
        features.loc[:, col] = label_encoder.fit_transform(features[col])
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    features = features.values.astype(np.float32)
    labels = labels.astype(np.int32)

    return features, labels


def normalization(data):
    data_min = np.amin(data, axis=0)
    data_max = np.amax(data, axis=0)
    data = (data - data_min) / (data_max - data_min)
    return data


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def balance_two_class(features, labels):
    yes_indices = np.where(labels == 1)[0]
    no_indices = np.where(labels == 0)[0]
    random_no_indices = np.random.choice(no_indices, size=len(yes_indices), replace=False)

    indices = np.concatenate((yes_indices, random_no_indices))
    np.random.shuffle(indices)

    balanced_features = features[indices]
    balanced_labels = labels[indices]

    return balanced_features, balanced_labels


def balance_under_sampling(features, labels):
    sampler = NearMiss()
    balanced_features, balanced_labels = sampler.fit_resample(features, labels)
    
    print(f"yes:no {np.sum(labels == 1)}:{np.sum(labels == 0)} -> {np.sum(balanced_labels == 1)}:{np.sum(balanced_labels == 0)}")

    return balanced_features, balanced_labels


def balance_over_sampling(features, labels):
    sampler = SMOTE(sampling_strategy="minority")
    balanced_features, balanced_labels = sampler.fit_resample(features, labels)
    
    print(f"yes:no {np.sum(labels == 1)}:{np.sum(labels == 0)} -> {np.sum(balanced_labels == 1)}:{np.sum(balanced_labels == 0)}")

    return balanced_features, balanced_labels


def handle_bank():
    features, labels = load_bank("./bank_additional/bank-additional-full.csv")
    features = normalization(features)
    np.save('./bank_data.npy', features)
    np.save('./bank_label.npy', labels)


def handle_under_sampling_bank():
    features, labels = load_bank("./bank_additional/bank-additional-full.csv")
    features = normalization(features)

    balanced_features, balanced_labels = balance_under_sampling(features, labels)
    balanced_features, balanced_labels = shuffle(balanced_features, balanced_labels, random_state=42)

    np.save('./bank_under_sampling_data.npy', balanced_features)
    np.save('./bank_under_sampling_label.npy', balanced_labels)


def handle_over_sampling_bank():
    features, labels = load_bank("./bank_additional/bank-additional-full.csv")
    features = normalization(features)

    features, labels = balance_over_sampling(features, labels)
    features, labels = shuffle(features, labels, random_state=42)

    np.save('./bank_over_sampling_data.npy', features)
    np.save('./bank_over_sampling_label.npy', labels)


def handle_credit():
    features, labels = load_credit("./credit_card_clients/UCI_Credit_Card.csv")
    features = normalization(features)

    np.save('./credit_data.npy', features)
    np.save('./credit_label.npy', labels)


def handle_under_sampling_credit():
    features, labels = load_credit("./credit_card_clients/UCI_Credit_Card.csv")
    features = normalization(features)

    balanced_features, balanced_labels = balance_under_sampling(features, labels)
    balanced_features, balanced_labels = shuffle(balanced_features, balanced_labels, random_state=42)

    np.save('./credit_under_sampling_data.npy', balanced_features)
    np.save('./credit_under_sampling_label.npy', balanced_labels)


def handle_over_sampling_credit():
    features, labels = load_credit("./credit_card_clients/UCI_Credit_Card.csv")
    features = normalization(features)

    features, labels = balance_over_sampling(features, labels)
    features, labels = shuffle(features, labels, random_state=42)

    np.save('./credit_over_sampling_data.npy', features)
    np.save('./credit_over_sampling_label.npy', labels)


def handle_digits():
    X, y = load_digits(return_X_y=True)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = np.expand_dims(X.reshape((-1, 8, 8)), 1)
    np.save('./digits_data.npy', X)
    np.save('./digits_label.npy', y)


def handle_car():
    features, labels = load_car("./car_evaluation/car.csv")
    # features = normalization(features)
    np.save('./car_data.npy', features)
    np.save('./car_label.npy', labels)


if __name__ == "__main__":
    handle_under_sampling_credit()
    handle_under_sampling_bank()
    handle_bank()
    handle_credit()
    handle_digits()
    handle_car()
    pass