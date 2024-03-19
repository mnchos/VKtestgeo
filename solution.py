import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# Загрузка тестовых данных
test_data = pd.read_csv("test_with_features.csv")

# Предобработка данных
scaler = StandardScaler()
X_test = scaler.fit_transform(test_data.drop(columns=['id']))

# Преобразование данных в тензоры PyTorch
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Загрузка обученной модели
model = torch.load("trained_model.pth")

# Оценка модели
with torch.no_grad():
    model.eval()
    predictions = model(X_test_tensor).squeeze().numpy()

# Сохранение предсказаний в файл
test_data['score'] = predictions
test_data[['id', 'score']].to_csv("submission.csv", index=False)

print("Предсказания сохранены в файл predictions.csv")