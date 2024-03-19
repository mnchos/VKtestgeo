import pandas as pd

from sklearn.neighbors import NearestNeighbors
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
features_data = pd.read_csv("features.csv")
# Выделение координат из train и test данных
train_coordinates = train_data[['lat', 'lon']].values
test_coordinates = test_data[['lat', 'lon']].values
# Обучение модели ближайших соседей на основе координат
nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(features_data[['lat', 'lon']])
# Поиск ближайших соседей для train и test данных
_, train_indices = nbrs.kneighbors(train_coordinates)
_, test_indices = nbrs.kneighbors(test_coordinates)
# Соединение признаков из features с train и test данными
train_features = features_data.iloc[train_indices.flatten()].reset_index(drop=True)
test_features = features_data.iloc[test_indices.flatten()].reset_index(drop=True)
# Добавление признаков к исходным данным
train_data = pd.concat([train_data, train_features.iloc[:, 2:]], axis=1)
test_data = pd.concat([test_data, test_features.iloc[:, 2:]], axis=1)  # Исключаем координаты из признаков тестового набора

# Сохранение новых данных в CSV файлы
train_data.to_csv("train_with_features.csv", index=False)
test_data.to_csv("test_with_features.csv", index=False)

print("Новые данные сохранены в файлы train_with_features.csv и test_with_features.csv")