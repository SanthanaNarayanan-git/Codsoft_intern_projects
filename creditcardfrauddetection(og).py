# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 15:42:44 2024

@author: Santhana narayanan
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

credit_card_data = pd.read_csv('creditcard.csv')

print(credit_card_data.head())

legid = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print(legid.Amount.describe())
print(fraud.Amount.describe())
print(credit_card_data.groupby('Class').mean())

legid_sample = legid.sample(n=492)

new_dataset = pd.concat([legid_sample, fraud], axis=0)

print(new_dataset['Class'].value_counts())
print(new_dataset.groupby('Class').mean())

X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=3)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=117)
model.fit(X_train_scaled, Y_train)

X_train_prediction = model.predict(X_train_scaled)
X_test_prediction = model.predict(X_test_scaled)

training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print(f"Training Data Accuracy: {training_data_accuracy}")
print(f"Test Data Accuracy: {test_data_accuracy}")

