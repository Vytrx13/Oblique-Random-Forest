import numpy as np
import pandas as pd

data = np.load('data.npz')
X_train = data['X_train']
y_train = data['y_train']

num_features = X_train.shape[1]
feature_names = [f'feature_{i+1}' for i in range(num_features)]

df = pd.DataFrame(X_train, columns=feature_names)
df['target'] = y_train

df.to_csv('train_data.csv', index=False)

print("Arquivo 'train_data.csv' gerado com sucesso.")