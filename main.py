from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Подготовка папок и загрузка датасета
Path("results").mkdir(exist_ok=True)

# Читаем файл локально из папки проекта (предварительно убедитесь, что файл winequality-red.csv находится рядом с main.py)
df = pd.read_csv("winequality-red.csv", sep=";")

print("Первые 5 строк датасета:")
print(df.head())

print("\nРазмер датасета (строки, столбцы):")
print(df.shape)

print("\nТипы данных:")
print(df.dtypes)

print("\nПропущенные значения:")
print(df.isnull().sum())


# 2. Выбираем признаки и целевую переменную
target = "quality"

X = df.drop(columns=[target])
y = df[target]

# Делим данные одинаково для обеих моделей
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 3. Базовая модель: все числовые признаки
model_all = LinearRegression()
model_all.fit(X_train, y_train)

pred_all = model_all.predict(X_test)


# 4. Проверяем мультиколлинеарность
corr = X_train.corr()

plt.figure(figsize=(12, 9))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Корреляционная матрица признаков")
plt.tight_layout()
plt.savefig("results/correlation_matrix.png")
plt.close()

# Если абсолютная корреляция двух признаков >= 0.75,
# удаляем один из них из сокращённого набора.
upper = corr.abs().where(
    pd.DataFrame(
        [[True if j > i else False for j in range(len(corr.columns))]
         for i in range(len(corr.columns))],
        index=corr.columns,
        columns=corr.columns
    )
)

to_drop = [
    column for column in upper.columns
    if (upper[column] >= 0.75).any()
]

selected_features = [col for col in X.columns if col not in to_drop]

print("\nПризнаки, удалённые из-за высокой корреляции:")
print(to_drop if to_drop else "Нет")

print("\nОставшиеся признаки:")
print(selected_features)


# 5. Обучаем модель на сокращённом наборе
model_reduced = LinearRegression()

model_reduced.fit(
    X_train[selected_features],
    y_train
)

pred_reduced = model_reduced.predict(
    X_test[selected_features]
)


# 6. Сравниваем метрики двух моделей
def calculate_metrics(y_true, predictions):
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": mean_squared_error(y_true, predictions) ** 0.5,
        "R2": r2_score(y_true, predictions)
    }


metrics = pd.DataFrame(
    {
        "Все признаки": calculate_metrics(y_test, pred_all),
        "Сокращённый набор": calculate_metrics(y_test, pred_reduced)
    }
).T

print("\nСравнение моделей:")
print(metrics.round(4))

metrics.to_csv("results/model_comparison.csv")


# 7. График остатков базовой модели
residuals = y_test - pred_all

plt.figure(figsize=(8, 5))
plt.scatter(pred_all, residuals, alpha=0.6)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Предсказанное качество")
plt.ylabel("Остатки (реальное - предсказанное)")
plt.title("График остатков базовой модели")
plt.tight_layout()
plt.savefig("results/residuals.png")
plt.close()


# 8. Сохраняем список признаков и итоговый вывод
with open("results/selected_features.txt", "w", encoding="utf-8") as file:
    file.write("Оставшиеся признаки:\n")
    file.write("\n".join(selected_features))
    file.write("\n\nУдалённые признаки:\n")
    file.write("\n".join(to_drop) if to_drop else "Нет")

print("\nФайлы с результатами сохранены в папке results.")
print("Практическая работа завершена.")
