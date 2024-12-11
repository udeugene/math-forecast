import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt

st.title("Линейная многофакторная модель (ЛМФМ)")

def apply_time_lags(data, lags):
    lagged_data = data.copy()
    for lag in lags:
        for col in lagged_data.columns:
            lagged_data[f"{col}_lag{lag}"] = lagged_data[col].shift(lag)
    lagged_data = lagged_data.dropna()
    return lagged_data

uploaded_file = st.file_uploader("Загрузите текстовый файл с данными", type=["csv", "xls", "xlsx", "ods"])
if uploaded_file:
    if uploaded_file.name.endswith("ods"):
        data = pd.read_excel(uploaded_file, engine="odf")
    else:
        data = pd.read_csv(uploaded_file)

    st.write("Предпросмотр данных:")
    st.dataframe(data.head())

    date_col = st.selectbox("Выберите столбец с датой:", options=data.columns, index=0)
    if date_col:
        data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
        if data[date_col].isnull().any():
            st.warning(f"Некоторые значения в столбце {date_col} не распознаны как даты и будут исключены.")
        data = data.dropna(subset=[date_col])
        data = data.sort_values(by=date_col)
        data["numeric_date"] = data[date_col].map(pd.Timestamp.toordinal)

    response_col = st.selectbox("Выберите столбец отклика (y):", options=[col for col in data.columns if col != date_col])
    predictors = st.multiselect("Выберите столбцы факторов (X):", options=[col for col in data.columns if col != response_col and col != date_col])

    if response_col and predictors:
        y = data[response_col]
        X = data[predictors]

        lags = st.multiselect("Выберите временные лаги для факторов:", options=list(range(1, 6)), default=[1])
        if lags:
            lagged_data = apply_time_lags(X, lags)
            X = lagged_data
            y = y.iloc[len(y) - len(X):]

        significance_level = st.slider("Выберите уровень значимости для отбора факторов", min_value=0.01, max_value=0.1, value=0.05, step=0.01)

        correlation_matrix = X.corr()
        st.write("Матрица корреляции между факторами:")
        st.dataframe(correlation_matrix)

        target_correlation = X.corrwith(y)
        st.write("Коэффициенты корреляции факторов с откликом:")
        st.dataframe(target_correlation)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_sm = sm.add_constant(X_train)
        ols_model = sm.OLS(y_train, X_train_sm).fit()

        st.write("Список факторов и их p-значения:")
        factors_pvalues = pd.DataFrame({"Фактор": X_train.columns, "p-value": ols_model.pvalues[1:]})
        for index, row in factors_pvalues.iterrows():
            factor = row["Фактор"]
            pvalue = row["p-value"]
            if pvalue < significance_level:
                st.markdown(f"<span style='color:green'>Фактор: {factor} значим на уровне значимости {significance_level} (p_value={pvalue:.3f})</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:red'>Фактор: {factor} не значим на уровне значимости {significance_level} (p_value={pvalue:.3f})</span>", unsafe_allow_html=True)
            keep = st.radio(f"Оставить фактор {factor}?", options=["Да", "Нет"], index=0)
            if keep == "Нет":
                X = X.drop(columns=[factor])

        st.write("Факторы после отбора:")
        st.write(X.columns.tolist())

        if st.button("Применить обновленные факторы и построить модель"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            lm = LinearRegression()
            lm.fit(X_train, y_train)
            y_pred = lm.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            residuals = y_test - y_pred
            mean_relative_error = np.mean(np.abs(residuals / y_test))

            f_stat = sm.OLS(y_train, sm.add_constant(X_train)).fit().fvalue
            f_pvalue = sm.OLS(y_train, sm.add_constant(X_train)).fit().f_pvalue

            st.subheader("Оценка адекватности модели")
            st.write(f"Среднеквадратическая ошибка (MSE): {mse:.2f}")
            st.write(f"Коэффициент детерминации (R²): {r2:.2f}")
            st.write(f"Средняя относительная ошибка (E): {mean_relative_error:.2%}")
            st.write(f"F-статистика модели: {f_stat:.2f}, p-value: {f_pvalue:.3f}")
            if f_pvalue < significance_level:
                st.success("Модель адекватна на заданном уровне значимости.")
            else:
                st.error("Модель неадекватна на заданном уровне значимости.")

            coef_df = pd.DataFrame({"Фактор": X_train.columns, "Коэффициент": lm.coef_})
            st.subheader("Коэффициенты модели")
            st.dataframe(coef_df)

        st.subheader("Прогнозирование")
        new_data = st.file_uploader("Загрузите файл с новыми факторами для прогнозирования", type=["csv", "xls", "xlsx", "ods"])
        if new_data:
            if new_data.name.endswith("ods"):
                new_X = pd.read_excel(new_data, engine="odf")
            else:
                new_X = pd.read_csv(new_data)

            st.write("Новые данные для прогнозирования:")
            st.dataframe(new_X)

            if date_col in new_X.columns:
                new_X[date_col] = pd.to_datetime(new_X[date_col], errors='coerce')
                new_X = new_X.dropna(subset=[date_col])
                new_X["numeric_date"] = new_X[date_col].map(pd.Timestamp.toordinal)

            new_X_filtered = new_X[X.columns]
            try:
                predictions = lm.predict(new_X_filtered)
                st.write("Результаты прогнозирования:")
                st.dataframe(pd.DataFrame({"Предсказания": predictions}))

                st.subheader("График прогноза, истинных значений и фактической истории")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(data["numeric_date"], y, label="Истинные значения (вся история)", color="blue")
                ax.plot(new_X["numeric_date"], predictions, label="Прогноз", color="red")
                ax.set_xlabel("Время")
                ax.set_ylabel(response_col)
                ax.legend()
                ax.grid()
                st.pyplot(fig)

            except NameError:
                st.error("Сначала создайте модель, нажав на кнопку 'Применить обновленные факторы и построить модель'.")
