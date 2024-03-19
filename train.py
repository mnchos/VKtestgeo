import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Загрузка данных
train_data = pd.read_csv("train_with_features.csv")

# Предобработка данных
scaler = StandardScaler()
X = scaler.fit_transform(train_data.drop(columns=['id', 'score']))
y = train_data['score'].values

# Разделение данных на тренировочную и валидационную выборки с сидированием
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразование данных в тензоры PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Установка сида для воспроизводимости результатов
torch.manual_seed(42)

# Определение архитектуры нейронной сети
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        self.dropou=nn.Dropout(0.25)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x= self.dropou(x)
        x = torch.relu(self.fc2(x))
        x= self.dropou(x)
        x = torch.relu(self.fc3(x))
        x= self.dropou(x)
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Инициализация модели
input_size = X_train.shape[1]
model = Net(input_size)

# Определение функции потерь и оптимизатора
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Обучение модели с early stopping
num_epochs = 100
batch_size = 64
early_stopping_epochs = 10  # количество эпох для early stopping
best_val_loss = float('inf')
patience = 0

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    val_mae = 0.0
    train_mae = 0.0

    # Обучение
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(inputs)
        train_mae += mean_absolute_error(targets.numpy(), outputs.squeeze().detach().numpy()) * len(inputs)
    train_loss /= len(train_loader.dataset)
    train_mae /= len(train_loader.dataset)

    # Валидация
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item() * len(inputs)
            val_mae += mean_absolute_error(targets.numpy(), outputs.squeeze().detach().numpy()) * len(inputs)
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience = 0
    else:
        patience += 1
        if patience >= early_stopping_epochs:
            print(f"Early stopping on epoch {epoch+1}")
            break

# Сохранение обученной модели
torch.save(model, "trained_model.pth")
print("Обученная модель сохранена в файл trained_model.pth")
