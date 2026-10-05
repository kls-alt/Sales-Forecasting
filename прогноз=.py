import pandas as pd
import numpy as np

# 1.создаем df
# 1.1. задаем генератор случайных чисел, но числа должны повторяться
np.random.seed(42)
# 1.2 создаем непрерывную последовательность из 180 дат (примерно полгода)
dates = pd.date_range(start='2026-01-01', periods=180, freq="D")
# 1.3 начинаем собирать sales с базового уровня в 100 штук
sales = []
for i, date in enumerate(dates):
    base = 100
    trend = i * 0.15 # плавный рост продаж со временем (+15% к лету)

    # если день суббота(5) или воскресенье(6) - увеличиваем продажи на 30%
    weekend_anomaly = 30 if date.dayofweek in [5, 6] else 0
    # делаем случайный шум (идеальных графиков не бывает)
    noise = np.random.randint(-15, 15)
    # итоговые продажи за день
    day_sales = int(base + trend + weekend_anomaly + noise)
    sales.append(day_sales)

# 1.4 собираем все в один большой df
df = pd.DataFrame({"date": dates, "sales": sales})
# просмотрим табличку
print(f"Размер таблицы: {df.shape} (180 строк, 2 колонки)\n")
print("Первые 5 строк датасета:")
print(df.head())

# 2. делаем признаки
# 2.1 извлекаем полезные фичи
df["day_of_week"] = df["date"].dt.dayofweek # 0-понедельник, 6-воскресенье
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
# 2.2 добавляем историю: продажи вчера(Lag_1) и продажи неделю назад(Lag_7)
df["sales_yesterday"] = df["sales"].shift(1)
df["sales_week_ago"] = df["sales"].shift(7)
# 2.3 добавляем скользящее среднее за 7 дней (средний спрос за неделю)
df["rolling_mean_7d"] = df["sales"].shift(7).rolling(window=7).mean()
# 2.4 убираем пустоту для первых строчек
df = df.dropna().reset_index(drop=True)

print("\nТаблица после подготовки фич (показаны последние 5 строк):")
print(
    df[
        [
            "date",
            "sales",
            "is_weekend",
            "sales_yesterday",
            "sales_week_ago",
            "rolling_mean_7d",
        ]
    ].tail()
)

# 3. обучаем модель
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 3.1 выделяем признаки (X) и целевую переменную (y)
features = ["day_of_week", "is_weekend", "sales_yesterday", "sales_week_ago", "rolling_mean_7d"]
X = df[features]
y = df["sales"]
# 3.2 хронологический сплит во избежание утечки данных
split_point = 150
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
# 3.3 создаем и обучаем модель
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# 3.4 делаем прогноз на тесте
predictions = model.predict(X_test)
# 3.5 считаем среднюю ошибку (MAE)
mae = mean_absolute_error(y_test, predictions)
print(f"\nМодель успешно обучена на {split_point} днях!")
print(f"Средняя ошибка (MAE) прогноза на тесте: {mae:.2f} шт.")

# baseline
baseline_pred = X_test["sales_yesterday"]
baseline_mae = mean_absolute_error(y_test, baseline_pred)
print(f"Baseline MAE: {baseline_mae:.2f}")

# визуализация
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.plot(df.loc[y_test.index, "date"], y_test, label="Реальные")
plt.plot(df.loc[y_test.index, "date"], predictions, label="Прогноз")

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.show()

# важность признаков
importance = pd.Series(
    model.feature_importances_, 
    index=X_train.columns
).sort_values(ascending=False)
print(importance)


# делаем предсказание 
# 1. берем последнюю дату из датасета
last_date = df['date'].max()
# 2. создаем новые даты на 7 дней вперед
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7, freq='D')
# 3. делаем копию, чтобы не испортить оригинал
# и оставляем только нужные колонки для истории
predict_df = df[['date', 'sales', 'day_of_week', 'is_weekend']].copy()
# 4. добавляем 7 дней в таблицу, нан там, где ничего нет
for new_date in future_dates:
    new_row = {
        'date': new_date,
        'sales': np.nan,
        'day_of_week': new_date.dayofweek,
        'is_weekend': int(new_date.dayofweek in [5, 6])
    }
    predict_df = pd.concat([predict_df, pd.DataFrame([new_row])], ignore_index=True)
# 5. предсказываем, начиная с первой строчки
start_future_index = len(df)
for i in range(start_future_index, len(predict_df)):
    # считаем признаки для текущего будущ дня на основе прошлого
    sales_yesterday = predict_df.loc[i-1, 'sales']
    # продажи неделю назад (i-7, на 7 дней назад)
    sales_week_ago = predict_df.loc[i-7, 'sales']
    # скользящее среднее (7 последних известных дней)
    rolling_mean_7d = predict_df.loc[i-7:i-1, 'sales'].mean()
    # собраем все признаки в один вектор для модели в том же порядке
    current_features = pd.DataFrame([{
        'day_of_week': predict_df.loc[i, 'day_of_week'],
        'is_weekend': predict_df.loc[i, 'is_weekend'],
        'sales_yesterday': sales_yesterday,
        'sales_week_ago': sales_week_ago,
        'rolling_mean_7d': rolling_mean_7d
    }])

    # модель делает предсказание на этот один день
    predict_sales = model.predict(current_features)[0]

    # предсказанное значение продаж записываем и заполняем нан
    predict_df.loc[i, 'sales'] = int(predict_sales)

# 6. результат
print("Прогноз продаж на будущую неделю:")
print(predict_df.tail(7)[['date', 'day_of_week', 'sales']])