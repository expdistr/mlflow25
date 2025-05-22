import mlflow
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Загрузка данных
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

# Настройка MLFlow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Diabetes Prediction")

# Начало эксперимента
with mlflow.start_run():
    # Параметры модели
    alpha = 0.5
    
    # Логируем параметры
    mlflow.log_param("alpha", alpha)
    
    # Обучение модели
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    
    # Предсказание и оценка
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    # Логируем метрики
    mlflow.log_metric("mse", mse)
    
    # Сохраняем модель
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Model trained with MSE: {mse}")