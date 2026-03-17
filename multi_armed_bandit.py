import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Подготовка среды (на базе Iris)
data = load_iris()
X = StandardScaler().fit_transform(data.data)
y = data.target

n_features = X.shape[1]
n_actions = len(np.unique(y))

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 2. Модель MLP
class BanditNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BanditNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

# 3. Агент
device = torch.device("cpu")
model = BanditNetwork(n_features, n_actions).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

epsilon = 0.1  # Шанс случайного выбора (exploration)
epsilon_decay = 0.995
epsilon_min = 0.05
epochs = 500
losses = []
correct_predictions_train = 0

# 4. Цикл обучения
for epoch in range(epochs):
    # Выбираем случайный индекс из обучающей выборки (имитация прихода контекста)
    idx = np.random.randint(0, len(X_train))
    context = torch.FloatTensor(X_train[idx]).unsqueeze(0).to(device)
    true_label = y_train[idx]

    # Выбор действия (Epsilon-Greedy)
    if np.random.rand() < epsilon:
        action = np.random.randint(0, n_actions)
    else:
        with torch.no_grad():
            q_values = model(context)
            action = torch.argmax(q_values).item()

    # Получаем награду
    reward = 1.0 if action == true_label else 0.0
    
    # Обучение
    target_q = model(context)
    # Мы хотим, чтобы модель предсказывала reward для выбранного действия
    target_f = target_q.clone().detach()
    target_f[0][action] = reward
    
    optimizer.zero_grad()
    loss = criterion(model(context), target_f)
    loss.backward()
    optimizer.step()

    # Постепенно затухаем epsilon, но не опускаемся ниже нижней границы
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    losses.append(loss.item())
    if reward == 1.0:
        correct_predictions_train += 1

# 5. Оценка на тестовой выборке (без exploration)
model.eval()
with torch.no_grad():
    test_contexts = torch.FloatTensor(X_test).to(device)
    test_labels = torch.LongTensor(y_test).to(device)
    test_q_values = model(test_contexts)
    test_actions = torch.argmax(test_q_values, dim=1)
    test_accuracy = (test_actions == test_labels).float().mean().item()

print(f"Train accuracy (online) после {epochs} итераций: {correct_predictions_train/epochs:.2%}")
print(f"Test accuracy: {test_accuracy:.2%}")
print(f"Финальный epsilon: {epsilon:.4f}")