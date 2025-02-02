import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import norm, chi2

# Configurar el puerto de Streamlit basado en la variable de entorno
port = int(os.environ.get("PORT", 8080))
st.set_page_config(page_title="VaR App")

# Estilo de la tabla
def highlight_sum(row):
    if row['Ticker'] == 'Sum de VaR Indiv.':
        return ['background-color: #778899'] * len(row)
    return [''] * len(row)


# Configuración de la aplicación
st.title('Aplicación del VaR Paramétrico')
st.subheader('Comparación con diferentes metodologías')
st.sidebar.header('Parámetros de Entrada')

# Configurar el estilo de Seaborn
sns.set(style="whitegrid")

# Selección de tickers
tickers_mexicanos = [
    'AMXL.MX', 'WALMEX.MX', 'CEMEXCPO.MX', 'BIMBOA.MX', 'GFNORTEO.MX',
    'ELEKTRA.MX', 'ALFAA.MX', 'GMEXICOB.MX', 'TLEVISACPO.MX', 'FEMSAUBD.MX',
    'PE&OLES.MX', 'GCC.MX', 'AMXB.MX', 'LIVEPOLC-1.MX', 'GFINBURO.MX'
]
selected_tickers = st.sidebar.multiselect('Selecciona hasta 5 acciones',
                                          tickers_mexicanos,
                                          default=tickers_mexicanos[:5])

# Selección del rango de fechas
start_date = st.sidebar.date_input('Fecha de inicio',
                                   pd.to_datetime('2017-09-22'))
end_date = st.sidebar.date_input('Fecha de fin', pd.to_datetime('2018-09-24'))

# Ingreso de los pesos para cada ticker
weights = []
if selected_tickers:
    default_weight = 1.0 / len(selected_tickers)
    for ticker in selected_tickers:
        weight = st.sidebar.number_input(f'Peso para {ticker}',
                                         min_value=0.0,
                                         max_value=1.0,
                                         value=default_weight)
        weights.append(weight)

# Recalcular la suma de los pesos
total_weight = sum(weights)

# Validación de que la suma de los pesos sea igual a 1
if not np.isclose(total_weight, 1.0):
    st.sidebar.error('La suma de los pesos debe ser igual a 1.0')

# Ingreso del monto del portafolio
portfolio_value = st.sidebar.number_input('Monto del portafolio (MXN)',
                                          min_value=0.0,
                                          value=500000.0)

# Selección del nivel de confianza
confidence_level = st.sidebar.selectbox('Nivel de confianza para el VaR',
                                        [0.95, 0.99])

# Selección del horizonte de tiempo
horizonte_dias = st.sidebar.number_input('Horizonte de tiempo (días)',
                                         min_value=1,
                                         max_value=365,
                                         value=1)

# Selección del número de simulaciones de Monte Carlo
num_simulaciones = st.sidebar.number_input(
    'Número de simulaciones para el método Monte Carlo:',
    min_value=1000,
    max_value=100000,
    value=10000)

# Definir fechas dinámicas para el backtesting basadas en el periodo de entrenamiento
from datetime import timedelta

# Fecha de inicio del backtesting (día siguiente a la fecha de fin del entrenamiento)
backtest_start_date_default = end_date + timedelta(days=1)

# Fecha de fin del backtesting (un año después de la fecha de fin del entrenamiento)
backtest_end_date_default = end_date + timedelta(days=365)

# Mostrar las fechas en la barra lateral
st.sidebar.header('Parámetros de Backtesting')

backtest_start_date = st.sidebar.date_input('Fecha de inicio del Backtesting',
                                            backtest_start_date_default)
backtest_end_date = st.sidebar.date_input('Fecha de fin del Backtesting',
                                          backtest_end_date_default)

# Botón para cargar datos
if st.sidebar.button('Cargar datos'):
    if len(selected_tickers) > 5:
        st.error('Por favor, selecciona un máximo de 5 acciones.')
    elif not np.isclose(total_weight, 1.0):
        st.error('La suma de los pesos debe ser igual a 1.0')
    # Validadores de fechas del backtesting
    elif backtest_start_date <= end_date:
        st.error(
            'La fecha de inicio del backtesting debe ser posterior a la fecha de fin del periodo de entrenamiento.'
        )
    elif backtest_end_date <= backtest_start_date:
        st.error(
            'La fecha de fin del backtesting debe ser posterior a la fecha de inicio del backtesting.'
        )
    else:
        try:
            # Descarga de datos de Yahoo Finance
            data = yf.download(selected_tickers,
                               start=start_date,
                               end=end_date)['Adj Close']

            # Verificar si hay datos faltantes
            if data.isnull().values.any():
                st.warning(
                    'Hay datos faltantes en los precios descargados. Se eliminarán las filas con datos faltantes.'
                )
                data = data.dropna()

            # Agregar un separador horizontal
            st.divider()

            # Mostrar los datos
            st.subheader('Datos descargados')
            st.write(data)

            # Calcular rendimientos logarítmicos diarios
            daily_returns = np.log(data / data.shift(1)).dropna()

            # Calcular la media, desviación estándar y varianza de los rendimientos diarios
            mean_returns = daily_returns.mean()
            std_returns = daily_returns.std()
            var_returns = daily_returns.var()

            # Mostrar rendimientos diarios
            st.subheader('Rendimientos diarios')
            st.write(daily_returns)

            # Calcular la matriz de covarianza
            covariance_matrix = daily_returns.cov()

            # Crear una copia de la matriz de covarianza en porcentaje para mostrar
            covariance_matrix_percent = covariance_matrix * 100

            # Mostrar la matriz de varianzas y covarianzas
            st.subheader('Matriz de Varianzas y Covarianzas')
            st.write(covariance_matrix_percent.style.format('{:.4f}'))

            # Calcular el rendimiento ponderado del portafolio
            portfolio_returns = daily_returns.dot(weights)

            # Convertir los pesos a una Serie con el índice de los tickers seleccionados
            weights_series = pd.Series(weights, index=selected_tickers)

            # Calcular el VaR paramétrico individual
            var_level = 1 - confidence_level  # Convertir el nivel de confianza a nivel de significancia
            z_value = np.abs(norm.ppf(var_level))

            var_parametric_individual = z_value * std_returns * np.sqrt(
                horizonte_dias)

            # Calcular el VaR monetario individual
            var_monetary_individual = var_parametric_individual[
                selected_tickers] * portfolio_value * weights_series

            # Preparar los resultados para mostrar en la tabla
            var_results = pd.DataFrame({
                'Ticker':
                selected_tickers,
                'μ':
                mean_returns[selected_tickers].values,
                'σ':
                std_returns[selected_tickers].values,
                'σ²':
                var_returns[selected_tickers].values,
                'VaR (%)':
                var_parametric_individual[selected_tickers].values * 100,
                'Peso': (np.array(weights) * 100).round(2).astype(str) + '%',
                f'VaR Monetario ({portfolio_value} MXN)':
                var_monetary_individual[selected_tickers].values.round(2)
            })

            # Calcular las sumas de VaR individuales
            var_sum = pd.DataFrame({
                'Ticker': ['Sum de VaR Indiv.'],
                'μ': [None],
                'σ': [None],
                'σ²': [None],
                'VaR (%)':
                [var_parametric_individual[selected_tickers].sum() * 100],
                'Peso': ['100%'],
                f'VaR Monetario ({portfolio_value} MXN)':
                [var_monetary_individual[selected_tickers].sum().round(2)]
            })

            var_results = pd.concat([var_results, var_sum], ignore_index=True)

            # Definir el nombre de la columna
            var_monetary_column = f'VaR Monetario ({portfolio_value} MXN)'

            # Mostrar el VaR paramétrico individual
            st.subheader('VaR Paramétrico Individual')
            st.write(
                var_results.style.apply(highlight_sum, axis=1).format({
                    'μ':
                    '{:.6f}',
                    'σ':
                    '{:.6f}',
                    'σ²':
                    '{:.6f}',
                    'VaR (%)':
                    '{:.2f}',
                    var_monetary_column:
                    '{:,.2f}'
                }))

            # Calcular el VaR paramétrico del portafolio
            weighted_cov_matrix = np.dot(
                np.dot(np.array(weights), covariance_matrix),
                np.array(weights))

            # Volatilidad del portafolio
            portfolio_volatility = np.sqrt(weighted_cov_matrix)

            # Multiplicador del horizonte temporal
            horizon_multiplier = np.sqrt(horizonte_dias)

            # VaR paramétrico del portafolio
            var_parametric_portfolio = portfolio_value * z_value * portfolio_volatility * horizon_multiplier

            # Crear la tabla de resultados
            table_data = {
                '': [
                    'Inversión', '[wi]ᵗ[Σ][wi]', 'Volatilidad (σ)',
                    'Estadístico Z', '√horizonte t', 'VaR Port'
                ],
                'Valores': [
                    portfolio_value, weighted_cov_matrix, portfolio_volatility,
                    z_value, horizon_multiplier, var_parametric_portfolio
                ]
            }

            # Convertir a DataFrame para mostrarlo como tabla
            table_df = pd.DataFrame(table_data).set_index('')

            # función de formato personalizada (definición)
            def format_valor(row):
                if row.name in ['Inversión', 'VaR Port']:
                    return f"{row['Valores']:,.2f}"
                else:
                    return f"{row['Valores']:,.4f}"

            # Aplicar la función de formato a la columna 'Valores'
            table_df['Valores'] = table_df.apply(format_valor, axis=1)

            # Mostrar el detalle del VaR paramétrico del portafolio
            st.subheader('Detalle del VaR Paramétrico del Portafolio')
            st.write(table_df)

            # Mostrar el VaR paramétrico del portafolio
            st.subheader('VaR Paramétrico del Portafolio')
            st.write(f'{var_parametric_portfolio:,.2f} MXN')

            # Comparar la suma de los VaR individuales con el VaR del portafolio
            st.subheader(
                'Comparación del VaR No Diversificado con el VaR del Portafolio'
            )
            if var_monetary_individual.sum() > var_parametric_portfolio:
                st.write(
                    'La suma de los VaR Paramétrico individuales es mayor que el VaR Paramétrico del portafolio, indicando un efecto de diversificación.'
                )
            else:
                st.write(
                    'La suma de los VaR individuales es menor o igual que el VaR Paramétrico del portafolio.'
                )

            # Visualización del Histograma de Rendimientos
            st.header('Histograma de Rendimientos del Portafolio')
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(portfolio_returns,
                    bins=50,
                    alpha=0.75,
                    color='blue',
                    edgecolor='black')
            ax.set_title('Histograma de Rendimientos del Portafolio')
            ax.set_xlabel('Rendimientos')
            ax.set_ylabel('Frecuencia')
            st.pyplot(fig)

            # Agregar un separador horizontal
            st.divider()

            # Calcular el VaR histórico individual
            #var_historical_individual = daily_returns.apply(lambda x: np.percentile(x, var_level * 100))
            var_historical_individual = daily_returns.quantile(
                var_level, axis=0)  #usando pandas

            # Calcular el VaR monetario histórico individual
            var_monetary_historical_individual = var_historical_individual[
                selected_tickers] * portfolio_value * weights_series

            # Preparar los resultados para mostrar en la tabla
            var_historical_results = pd.DataFrame({
                'Ticker':
                selected_tickers,
                'VaR Histórico (%)':
                var_historical_individual[selected_tickers].values * 100,
                f'VaR Monetario Histórico ({portfolio_value} MXN)':
                var_monetary_historical_individual[selected_tickers].values.
                round(2)
            })

            # Calcular las sumas de VaR históricos individuales
            var_historical_sum = pd.DataFrame({
                'Ticker': ['Sum de VaR Histórico Indiv.'],
                'VaR Histórico (%)':
                [var_historical_individual[selected_tickers].sum() * 100],
                f'VaR Monetario Histórico ({portfolio_value} MXN)': [
                    var_monetary_historical_individual[selected_tickers].sum().
                    round(2)
                ]
            })

            var_historical_results = pd.concat(
                [var_historical_results, var_historical_sum],
                ignore_index=True)

            # Mostrar el VaR histórico individual
            st.subheader('VaR Histórico Individual')
            st.write(
                var_historical_results.style.apply(
                    highlight_sum, axis=1).format({
                        'VaR Histórico (%)':
                        '{:.2f}',
                        f'VaR Monetario Histórico ({portfolio_value} MXN)':
                        '{:,.2f}'
                    }))

            # Calcular el VaR histórico del portafolio
            historical_losses = portfolio_returns.sort_values()
            var_historical_portfolio = historical_losses.quantile(
                var_level) * portfolio_value * -1

            # Mostrar el VaR histórico del portafolio
            st.subheader('VaR Histórico del Portafolio')
            st.write(f'{var_historical_portfolio:,.2f} MXN')

            # Comparar la suma de los VaR individuales con el VaR del portafolio (Histórico)
            st.subheader(
                'Comparación del VaR No Diversificado con el VaR Histórico del Portafolio'
            )
            if abs(var_monetary_historical_individual.sum()) > abs(
                    var_historical_portfolio):
                st.write(
                    'La suma de los VaR Históricos individuales es mayor que el VaR Histórico del portafolio, indicando un efecto de diversificación.'
                )
            else:
                st.write(
                    'La suma de los VaR individuales es menor o igual que el VaR Histórico del portafolio.'
                )

            # Agregar un separador horizontal
            st.divider()

            # Calcular el CVaR histórico individual
            cvar_historical_individual = daily_returns.apply(
                lambda x: x[x <= np.percentile(x, var_level * 100)].mean())

            # Calcular el CVaR monetario histórico individual
            cvar_monetary_historical_individual = cvar_historical_individual[
                selected_tickers] * portfolio_value * np.array(weights)

            # Preparar los resultados para mostrar en la tabla
            cvar_historical_results = pd.DataFrame({
                'Ticker':
                selected_tickers,
                'CVaR Histórico (%)':
                cvar_historical_individual[selected_tickers].values * 100,
                f'CVaR Monetario Histórico ({portfolio_value} MXN)':
                cvar_monetary_historical_individual[selected_tickers].values.
                round(2)
            })

            # Calcular las sumas de CVaR históricos individuales
            cvar_historical_sum = pd.DataFrame({
                'Ticker': ['Sum de CVaR Histórico Indiv.'],
                'CVaR Histórico (%)':
                [cvar_historical_individual[selected_tickers].sum() * 100],
                f'CVaR Monetario Histórico ({portfolio_value} MXN)': [
                    cvar_monetary_historical_individual[selected_tickers].sum(
                    ).round(2)
                ]
            })

            cvar_historical_results = pd.concat(
                [cvar_historical_results, cvar_historical_sum],
                ignore_index=True)

            # Mostrar el CVaR histórico individual
            st.subheader('CVaR Histórico Individual')
            st.write(
                cvar_historical_results.style.apply(
                    highlight_sum, axis=1).format({
                        'CVaR Histórico (%)':
                        '{:.2f}',
                        f'CVaR Monetario Histórico ({portfolio_value} MXN)':
                        '{:,.2f}'
                    }))

            # Calcular el CVaR histórico del portafolio
            cvar_historical_portfolio = historical_losses[
                historical_losses <= historical_losses.quantile(
                    var_level)].mean() * portfolio_value * -1

            # Mostrar el CVaR histórico del portafolio
            st.subheader('CVaR Histórico del Portafolio')
            st.write(f'{cvar_historical_portfolio:,.2f} MXN')

            # Comparar la suma de los CVaR individuales con el CVaR del portafolio
            st.subheader(
                'Comparación del CVaR No Diversificado con el CVaR del Portafolio'
            )
            if abs(cvar_monetary_historical_individual.sum()) > abs(
                    cvar_historical_portfolio):
                st.write(
                    'La suma de los CVaR individuales es mayor que el CVaR del portafolio, indicando un efecto de diversificación.'
                )
            else:
                st.write(
                    'La suma de los CVaR individuales es menor o igual que el CVaR del portafolio.'
                )

            # Agregar un separador horizontal
            st.divider()

            # Por default el numero de simulaciones es de 10,000 y dias_simulacion es el horizonte de días para Monte Carlo
            dias_simulacion = horizonte_dias

            # Función para calcular el VaR utilizando el método Monte Carlo
            def calcular_var_montecarlo(daily_returns, data, selected_tickers,
                                        portfolio_value, weights,
                                        confidence_level, num_simulaciones,
                                        dias_simulacion):
                mean_returns = daily_returns.mean()
                cov_matrix = daily_returns.cov()

                # Mostrar parámetros utilizados
                st.subheader('Parámetros utilizados para Monte Carlo')
                # Convertir weights a un formato legible
                weights_str = ', '.join(
                    [f'{weight:.2%}' for weight in weights])
                parametros = {
                    'Portfolio Value': [portfolio_value],
                    'Confidence Level': [confidence_level],
                    'Num Simulaciones': [num_simulaciones],
                    'Horizonte (días)': [dias_simulacion],
                    'Weights': [weights_str],  # Incluir los pesos
                }
                parametros_df = pd.DataFrame(parametros)
                st.write(parametros_df)

                # Mostrar los precios iniciales de cada activo
                st.subheader('Precios Iniciales de los Activos Seleccionados')
                precios_iniciales = data.iloc[-1]
                st.write(
                    precios_iniciales.to_frame(
                        name='Precio Inicial').style.format('{:,.6f}'))

                # Mostrar la media y la volatilidad de los rendimientos de cada activo
                st.subheader('Estadísticas de Rendimientos Diarios')
                estadisticas_rendimientos = pd.DataFrame({
                    'Media de Rendimientos Diarios':
                    mean_returns,
                    'Volatilidad de Rendimientos Diarios':
                    daily_returns.std()
                })
                st.write(
                    estadisticas_rendimientos.style.format({
                        'Media de Rendimientos Diarios':
                        '{:.6f}',
                        'Volatilidad de Rendimientos Diarios':
                        '{:.6f}'
                    }))

                # Calcular la inversión inicial en cada activo
                inversion_inicial = portfolio_value * np.array(weights)

                # Calcular la cantidad de acciones de cada activo
                cantidad_acciones = inversion_inicial / data.iloc[-1].values

                # Generar todas las simulaciones de una vez
                rendimientos_simulados = np.random.multivariate_normal(
                    mean_returns, cov_matrix,
                    (num_simulaciones, dias_simulacion))

                # Calcular los precios simulados
                precios_simulados = data.iloc[-1].values * np.exp(
                    rendimientos_simulados.cumsum(axis=1))

                # Calcular el valor final de cada activo en cada simulación
                valor_activos_simulados = precios_simulados[:,
                                                            -1, :] * cantidad_acciones
                # Calcular las pérdidas individuales para cada activo
                perdidas_individuales = inversion_inicial - valor_activos_simulados

                # Calcular la media de los rendimientos simulados
                media_rendimientos_simulados = rendimientos_simulados.mean(
                    axis=(0, 1))
                st.subheader('Media de Rendimientos Simulados')
                media_simulados_df = pd.DataFrame(media_rendimientos_simulados,
                                                  index=selected_tickers,
                                                  columns=['Media Simulada'])
                st.write(media_simulados_df.style.format('{:.6f}'))

                # Mostrar ejemplos de precios simulados
                st.subheader('Ejemplos de Precios Simulados')
                num_ejemplos = 5  # Número de simulaciones a mostrar
                precios_simulados_df = pd.DataFrame(
                    precios_simulados[:num_ejemplos, :, :].reshape(
                        -1, len(selected_tickers)),
                    columns=selected_tickers)
                st.write(precios_simulados_df.head(10).style.format('{:,.2f}'))

                # Calcular el valor final del portafolio para cada simulación
                valor_portafolio_simulado = precios_simulados[:, -1, :].dot(
                    cantidad_acciones)

                # Mostrar ejemplos del valor final del portafolio en las simulaciones
                st.subheader('Valor Final del Portafolio simulado')
                valor_portafolio_simulado_df = pd.DataFrame(
                    valor_portafolio_simulado[:num_ejemplos],
                    columns=['Valor Final del Portafolio'])
                st.write(valor_portafolio_simulado_df.style.format('{:,.2f}'))

                # Calcular las pérdidas
                perdidas_simuladas = portfolio_value - valor_portafolio_simulado

                # Calcular el VaR del portafolio
                var_montecarlo = np.percentile(perdidas_simuladas,
                                               (1 - confidence_level) * 100)
                # Calcular el VaR individual para cada activo
                var_montecarlo_individual = np.percentile(
                    perdidas_individuales, (1 - confidence_level) * 100,
                    axis=0)

                # Mostrar la distribución de pérdidas simuladas
                st.subheader(
                    'Distribución de Pérdidas Simuladas - Monte Carlo')
                fig_mc, ax_mc = plt.subplots(figsize=(10, 6))
                ax_mc.hist(perdidas_simuladas,
                           bins=50,
                           alpha=0.75,
                           color='green',
                           edgecolor='black')
                ax_mc.axvline(var_montecarlo,
                              color='r',
                              linestyle='dashed',
                              linewidth=2)
                ax_mc.set_title('Histograma de Pérdidas Simuladas')
                ax_mc.set_xlabel('Pérdida')
                ax_mc.set_ylabel('Frecuencia')
                st.pyplot(fig_mc)

                # Mostrar histogramas de pérdidas individuales
                st.subheader(
                    'Distribución de Pérdidas Simuladas Individuales - Monte Carlo'
                )
                for i, ticker in enumerate(selected_tickers):
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.hist(perdidas_individuales[:, i],
                            bins=50,
                            alpha=0.75,
                            color='blue',
                            edgecolor='black')
                    ax.axvline(var_montecarlo_individual[i],
                               color='r',
                               linestyle='dashed',
                               linewidth=2)
                    ax.set_title(
                        f'Histograma de Pérdidas Simuladas - {ticker}')
                    ax.set_xlabel('Pérdida')
                    ax.set_ylabel('Frecuencia')
                    st.pyplot(fig)

                return var_montecarlo, var_montecarlo_individual

            # Cálculo del VaR Monte Carlo
            var_montecarlo, var_montecarlo_individual = calcular_var_montecarlo(
                daily_returns, data, selected_tickers, portfolio_value,
                weights, confidence_level, num_simulaciones, dias_simulacion)

            # Preparar los resultados para mostrar en la tabla
            var_montecarlo_results = pd.DataFrame({
                'Ticker':
                selected_tickers,
                'VaR Monte Carlo Individual (MXN)':
                var_montecarlo_individual.round(2)
            })

            # Calcular la suma de los VaR individuales
            var_montecarlo_sum = pd.DataFrame({
                'Ticker': ['Sum de VaR MC Indiv.'],
                'VaR Monte Carlo Individual (MXN)':
                [var_montecarlo_individual.sum().round(2)]
            })

            # Concatenar los resultados
            var_montecarlo_results = pd.concat(
                [var_montecarlo_results, var_montecarlo_sum],
                ignore_index=True)

            # Mostrar el VaR Monte Carlo Individual
            st.subheader('VaR Monte Carlo Individual')
            st.write(
                var_montecarlo_results.style.apply(
                    highlight_sum, axis=1).format(
                        {'VaR Monte Carlo Individual (MXN)': '{:,.2f}'}))

            # Mostrar el VaR Monte Carlo en la aplicación
            st.subheader('VaR utilizando el método Monte Carlo')
            st.write(f'{var_montecarlo:,.2f} MXN')

            # Agregar un separador horizontal
            st.divider()

            # Comparar la suma de los VaR individuales con el VaR del portafolio (Monte Carlo)
            st.subheader(
                'Comparación del VaR No Diversificado con el VaR del Portafolio (Monte Carlo)'
            )
            if abs(var_montecarlo_individual.sum()) > abs(var_montecarlo):
                st.write(
                    'La suma de los VaR Monte Carlo individuales es mayor que el VaR Monte Carlo del portafolio, indicando un efecto de diversificación.'
                )
            else:
                st.write(
                    'La suma de los VaR Monte Carlo individuales es menor o igual que el VaR Monte Carlo del portafolio.'
                )

            # Agregar un separador horizontal
            st.divider()

            # Crear un DataFrame para la comparación de metodologías
            comparacion_var = pd.DataFrame({
                'Metodología': [
                    'VaR Paramétrico', 'VaR Histórico', 'CVaR Histórico',
                    'VaR Monte Carlo'
                ],
                'VaR Individual (Sum)': [
                    abs(var_monetary_individual.sum()),
                    abs(var_monetary_historical_individual.sum()),
                    abs(cvar_monetary_historical_individual.sum()),
                    abs(var_montecarlo_individual.sum())
                ],
                'VaR del Portafolio': [
                    abs(var_parametric_portfolio),
                    abs(var_historical_portfolio),
                    abs(cvar_historical_portfolio),
                    abs(var_montecarlo)
                ]
            })

            # Calcular el efecto de diversificación
            comparacion_var['Efecto Diversificación'] = comparacion_var[
                'VaR Individual (Sum)'] - comparacion_var['VaR del Portafolio']

            # Mostrar la tabla
            st.subheader('Comparación de Metodologías de VaR')
            st.write(
                comparacion_var.style.format({
                    'VaR Individual (Sum)':
                    '{:,.2f}',
                    'VaR del Portafolio':
                    '{:,.2f}',
                    'Efecto Diversificación':
                    '{:,.2f}'
                }))

            # Crear un DataFrame con los VaR individuales por activo y metodología
            var_individuales = pd.DataFrame({
                'Ticker':
                selected_tickers,
                'VaR Paramétrico':
                var_monetary_individual.values,
                'VaR Histórico':
                var_monetary_historical_individual.values,
                'CVaR Histórico':
                cvar_monetary_historical_individual.values,
                'VaR Monte Carlo':
                var_montecarlo_individual
            })

            # Asegurarse de que los VaR sean valores positivos
            var_individuales[[
                'VaR Paramétrico', 'VaR Histórico', 'CVaR Histórico',
                'VaR Monte Carlo'
            ]] = var_individuales[[
                'VaR Paramétrico', 'VaR Histórico', 'CVaR Histórico',
                'VaR Monte Carlo'
            ]].abs()

            # Establecer 'Ticker' como índice
            var_individuales.set_index('Ticker', inplace=True)

            # Crear una figura para el heatmap
            st.subheader('Heatmap de VaR Individual por Metodología')
            fig, ax = plt.subplots(figsize=(10, 6))

            # Crear el heatmap
            sns.heatmap(var_individuales,
                        annot=True,
                        fmt=".2f",
                        cmap="YlGnBu",
                        ax=ax)

            # Ajustar las etiquetas y el título
            ax.set_title('VaR Individual por Activo y Metodología')
            ax.set_xlabel('Metodología')
            ax.set_ylabel('Activo')

            # Ajustar la rotación de las etiquetas en el eje X si es necesario
            plt.xticks(rotation=45)

            # Mostrar el heatmap en Streamlit
            st.pyplot(fig)

            # Agregar un separador horizontal
            st.divider()
            st.title('Backtesting')

            var_methods = {
                'VaR Paramétrico': var_parametric_portfolio,
                'VaR Histórico': var_historical_portfolio,
                'CVaR Histórico': cvar_historical_portfolio,
                'VaR Monte Carlo': abs(var_montecarlo)
            }

            # Descargar datos para el periodo de backtesting
            backtest_data = yf.download(selected_tickers,
                                        start=backtest_start_date,
                                        end=backtest_end_date)['Adj Close']

            # Verificar si hay datos faltantes en los datos de backtesting
            if backtest_data.isnull().values.any():
                st.warning(
                    'Hay datos faltantes en los precios descargados para el periodo de backtesting. Se eliminarán las filas con datos faltantes.'
                )
                backtest_data = backtest_data.dropna()

            # Mostrar los datos
            st.subheader('Datos descargados para el periodo de backtesting')
            st.write(backtest_data)

            # Calcular rendimientos logarítmicos diarios para el periodo de backtesting
            backtest_daily_returns = np.log(backtest_data /
                                            backtest_data.shift(1)).dropna()

            # Mostrar rendimientos logarítmicos diarios
            st.subheader('Rendimientos logarítmicos diarios')

            # Calcular el rendimiento del portafolio durante el periodo de backtesting
            backtest_portfolio_returns = backtest_daily_returns.dot(weights)

            # Agregar la columna del portafolio a la tabla de rendimientos diarios
            backtest_daily_returns['Portafolio'] = backtest_portfolio_returns

            # Mostrar los rendimientos logaritmicos
            st.write(backtest_daily_returns)

            # Calcular las pérdidas reales (en MXN) del portafolio durante el periodo de backtesting
            backtest_portfolio_values = [portfolio_value]
            for r in backtest_portfolio_returns:
                backtest_portfolio_values.append(
                    backtest_portfolio_values[-1] * np.exp(r))

            backtest_portfolio_values = np.array(backtest_portfolio_values)

            #Calcular las pérdidas diarias
            backtest_losses = np.diff(backtest_portfolio_values) * -1

            # Crear un DataFrame para los valores del portafolio con las fechas
            backtest_portfolio_df = pd.DataFrame({
                'Fecha':
                backtest_portfolio_returns.
                index,  # Usamos las fechas de los rendimientos del portafolio
                'Valor del Portafolio':
                backtest_portfolio_values[
                    1:]  # Quitamos el primer valor de la lista (500,000)
            })

            # Establecer la fecha como índice
            backtest_portfolio_df.set_index('Fecha', inplace=True)

            # Mostrar los valores del portafolio durante el backtesting con las fechas correspondientes
            st.subheader('Valores del Portafolio durante el Backtesting')
            st.write(backtest_portfolio_df)

            # Calcular excedencias para cada metodología
            metodologia_list = []
            excedencias_list = []
            porcentaje_excedencias_list = []

            for method, var_value in var_methods.items():
                excedencias = backtest_losses > var_value
                num_excedencias = excedencias.sum()
                num_dias = len(backtest_losses)
                porcentaje_excedencias = (num_excedencias / num_dias) * 100

                metodologia_list.append(method)
                excedencias_list.append(num_excedencias)
                porcentaje_excedencias_list.append(porcentaje_excedencias)

            # Crear un DataFrame con los resultados
            backtest_summary = pd.DataFrame({
                'Metodología':
                metodologia_list,
                'Número de observaciones':
                num_dias,
                'Número de Excedencias':
                excedencias_list,
                'Porcentaje de Excedencias (%)':
                porcentaje_excedencias_list
            })

            # Mostrar la tabla de excedencias
            st.subheader('Resultados del Backtesting por Metodología de VaR')
            st.write(
                backtest_summary.style.format({
                    'Número de observaciones':
                    '{:,.0f}',
                    'Número de Excedencias':
                    '{:,.0f}',
                    'Porcentaje de Excedencias (%)':
                    '{:.2f}%'
                }))

            # Crear un DataFrame para facilitar la visualización
            backtest_results = pd.DataFrame({
                'Fecha':
                backtest_portfolio_returns.index,
                'Pérdidas':
                backtest_losses
            })

            backtest_results.set_index('Fecha', inplace=True)

            # Gráfico de pérdidas y todos los VaR estimados
            st.subheader(
                'Pérdidas Reales vs. VaR Estimados de Todas las Metodologías')

            fig_bt, ax_bt = plt.subplots(figsize=(12, 8))

            # Graficar las pérdidas reales
            ax_bt.plot(backtest_results.index,
                       backtest_results['Pérdidas'],
                       label='Pérdida Real',
                       color='black')

            # Definir colores para cada metodología
            colors = {
                'VaR Paramétrico': 'blue',
                'VaR Histórico': 'orange',
                'CVaR Histórico': 'green',
                'VaR Monte Carlo': 'red'
            }

            # Graficar cada VaR
            for method, var_value in var_methods.items():
                ax_bt.axhline(y=var_value,
                              color=colors[method],
                              linestyle='--',
                              linewidth=2,
                              label=method)

            # Añadir título y etiquetas
            ax_bt.set_title('Backtesting del VaR con Todas las Metodologías')
            ax_bt.set_xlabel('Fecha')
            ax_bt.set_ylabel('Pérdida (MXN)')
            ax_bt.legend()

            # Mejorar la visualización
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Mostrar el gráfico en Streamlit
            st.pyplot(fig_bt)

        except Exception as e:
            st.error(f'Error al descargar o procesar los datos: {e}')
