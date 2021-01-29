import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import skew, kurtosis, chi2, linregress
from scipy.optimize import minimize
from numpy import linalg as LA
import altair as alt
import plotly.figure_factory as ff

#st.set_option('deprecation.showPyplotGlobalUse', False)
pd.set_option('precision', 4)



def load_timeseries(ric):
    path = 'Datos/'+ric+'.'+'csv'
    table_raw = pd.read_csv(path)
    t = pd.DataFrame()
    t['date'] = pd.to_datetime(table_raw['Date'], dayfirst=True)
    t['close'] = table_raw['Close']
    t.sort_values(by='date', ascending=True)
    t['close_previous'] = t['close'].shift(1)
    # rendimientos
    t['return_close'] = t['close']/t['close_previous'] - 1
    t = t.dropna()
    t = t.reset_index(drop=True)
    # entrada para el test de Jarque-Bera
    x = t['return_close'].values # rendimientos como arrays
    x_str = 'Rendimientos reales' + ric # label e.g. ric
    return x, x_str, t


def plot_timeseries_price(t, ric):
    plt.figure()
    plt.plot(t['date'],t['close'])
    plt.title('Serie de tiempo de los precios' + ric)
    plt.xlabel('Tiempo')
    plt.ylabel('Precio')
    plt.show()
    
    
def plot_histogram(x, x_str, plot_str, bins=100):
    # plot histograma
    plt.figure()
    plt.hist(x,bins)
    plt.title('Histograma ' + x_str)
    plt.xlabel(plot_str)
    plt.show()
    

def synchronise_timeseries(benchmark, ric):
    x1, str1, t1 = load_timeseries(benchmark)
    x2, str2, t2 = load_timeseries(ric)
    # sincronizando tiempos
    timestamp1 = list(t1['date'].values)
    timestamp2 = list(t2['date'].values)
    # hacemos la intersección de los tiempos
    timestamps = list(set(timestamp1) & set(timestamp2))
    # sincronizando serie de tiempo para x1 o ric
    t1_sync = t1[t1['date'].isin(timestamps)]
    t1_sync.sort_values(by='date', ascending=True)
    t1_sync = t1_sync.reset_index(drop=True)
    # sincronizando serie de tiempo para x2 o benchmark
    t2_sync = t2[t2['date'].isin(timestamps)]
    t2_sync.sort_values(by='date', ascending=True)
    t2_sync = t2_sync.reset_index(drop=True)
    # rendimientos del ric y benchmark
    t = pd.DataFrame()
    t['date'] = t1_sync['date']
    t['price_1'] = t1_sync['close'] # precio benchmark
    t['price_2'] = t2_sync['close'] # precio ric
    t['return_1'] = t1_sync['return_close'] # rendimiento benchmark
    t['return_2'] = t2_sync['return_close'] # rendimiento ric
    # calculamos los vectores de rendimientos
    returns_benchmark = t['return_1'].values # variable x
    returns_ric = t['return_2'].values # variable y
    return returns_benchmark, returns_ric, t # x, y, t


def compute_beta(benchmark, ric, bool_print=False):
    # calculamos la beta del modelo CAPM
    capm = capm_manager(benchmark, ric)
    capm.load_timeseries()
    capm.compute()
    if bool_print:
        print('------')
        print(capm)
    beta = capm.beta
    return beta

def compute_portfolio_min_variance(covariance_matrix, notional):
    eigenvalues, eigenvectors = LA.eigh(covariance_matrix)
    variance_explained = eigenvalues[0] / sum(abs(eigenvalues))
    eigenvector = eigenvectors[:,0]
    if max(eigenvector) < 0.0:
        eigenvector = - eigenvector
    port_min_variance = notional * eigenvector / sum(abs(eigenvector))
    return port_min_variance, variance_explained


def compute_portfolio_pca(covariance_matrix, notional):
    eigenvalues, eigenvectors = LA.eigh(covariance_matrix)
    variance_explained = eigenvalues[-1] / sum(abs(eigenvalues))
    eigenvector = eigenvectors[:,-1]
    if max(eigenvector) < 0.0:
        eigenvector = - eigenvector
    port_pca = notional * eigenvector / sum(abs(eigenvector))
    return port_pca, variance_explained


def compute_portfolio_equi_weight(size, notional):
    port_equi = (notional / size) * np.ones([size])
    return port_equi


def compute_portfolio_long_only(size, notional, covariance_matrix):
    # inicializamos la optimización
    x = np.zeros([size,1])
    # inicializamos las restricciones
    cons = [{"type": "eq", "fun": lambda x: sum(abs(x)) - 1}]
    bnds = [(0, None) for i in range(size)]
    # calculamos la optimización
    res = minimize(compute_portfolio_variance, x, args=(covariance_matrix), constraints=cons, bounds=bnds)
    port_long_only = notional * res.x
    return port_long_only


def compute_portfolio_markowitz(size, notional, covariance_matrix, returns, target_return):
    # inicializamos la optimización
    x = np.zeros([size,1])
    # inicializamos las restricciones
    cons = [{"type": "eq", "fun": lambda x: np.transpose(returns).dot(x).item() - target_return},\
            {"type": "eq", "fun": lambda x: sum(abs(x)) - 1}]
    bnds = [(0, None) for i in range(size)]
    # calculamos la optimización
    res = minimize(compute_portfolio_variance, x, args=(covariance_matrix), constraints=cons, bounds=bnds)
    weights = notional * res.x
    return weights


def compute_portfolio_variance(x, covariance_matrix):
    variance = np.dot(x.T, np.dot(covariance_matrix, x)).item()
    return variance


def compute_portfolio_volatility(covariance_matrix, weights):
    notional = sum(abs(weights))
    if notional <= 0.0:
        return 0.0
    weights = weights / notional # pesos unitarios en la norma L1
    variance = np.dot(weights.T, np.dot(covariance_matrix, weights)).item()
    if variance <= 0.0:
        return 0.0
    volatility = np.sqrt(variance)
    return volatility

class jarque_bera_test():
    
    def __init__(self, ric):
        self.ric = ric
        self.returns = []
        self.dataframe = pd.DataFrame()
        self.size = 0
        self.str_name = ''
        self.mean = 0.0
        self.std = 0.0
        self.skew = 0.0
        self.kurt = 0.0
        self.median = 0.0
        self.var_95 = 0.0
        self.cvar_95 = 0.0
        self.sharpe = 0.0
        self.jarque_bera = 0.0
        self.p_value = 0.0
        self.is_normal = 0.0
        
        
    def __str__(self):
        str_self = self.str_name + ' | Tamaño ' + str(self.size) + '\n' + self.plot_str()
        return str_self


    def load_timeseries(self):
        self.returns, self.str_name, self.dataframe = load_timeseries(self.ric)
        self.size = len(self.returns)

    def compute(self):
        self.mean = np.mean(self.returns)
        self.std = np.std(self.returns) # volatilidad
        self.skew = skew(self.returns)
        self.kurt = kurtosis(self.returns) # curtosis en exceso 
        # rendimientos entre volatilidad
        ########### tal vez se le debería de restar la rf, hay que checar eso
        self.sharpe = self.mean / (self.std * np.sqrt(252)) # anualizado (por eso lo dividimos entre la raíz de 252)
        self.median = np.median(self.returns)
        self.var_95 = np.percentile(self.returns,5)
        self.cvar_95 = np.mean(self.returns[self.returns <= self.var_95])
        # como es curtosis en exceso, no le restamos un 3 a la curtosis 
        self.jarque_bera = self.size/6*(self.skew**2 + 1/4*self.kurt**2)
        # los grados de libertad (df) es un dos por los resultadis del test de jarque-bera 
        self.p_value = 1 - chi2.cdf(self.jarque_bera, df=2)
        self.is_normal = (self.p_value > 0.05) # equivalentemente jb < 6


    def plot_str(self):
        nb_decimals = 4
        plot_str = 'Media ' + str(np.round(self.mean,nb_decimals))\
            + ' | Mediana ' + str(np.round(self.median,nb_decimals))\
            + ' | Desviación estándar ' + str(np.round(self.std,nb_decimals))\
            + ' | Coeficiente de asimetría ' + str(np.round(self.skew,nb_decimals)) + '\n'\
            + 'Curtosis ' + str(np.round(self.kurt,nb_decimals))\
            + ' | Sharpe ratio ' + str(np.round(self.sharpe,nb_decimals))\
            + ' | VaR 95% ' + str(np.round(self.var_95,nb_decimals))\
            + ' | CVaR 95% ' + str(np.round(self.cvar_95,nb_decimals)) + '\n'\
            + 'Test de Jarque-Bera ' + str(np.round(self.jarque_bera,nb_decimals))\
            + ' | p-value ' + str(np.round(self.p_value,nb_decimals))\
            + ' | ¿Es una distribución normal? ' + str(self.is_normal)
        return plot_str
    
    
    def plot_timeseries(self):
        plot_timeseries_price(self.dataframe, self.ric)
        
    
    def plot_histogram(self):
        plot_histogram(self.returns, self.str_name, self.plot_str())
    

###__________________________________________________________ CAPM ______________________________________________

class capm_manager():
    
    def __init__(self, benchmark, ric):
        self.ric = ric
        self.benchmark = benchmark
        self.returns_benchmark = [] # x
        self.returns_ric = [] # y
        self.dataframe = pd.DataFrame()
        self.alpha = None
        self.beta = None
        self.p_value = None
        self.null_hypothesis = False
        self.r_value = None
        self.r_squared = None
        self.correlation = None
        self.predictor_linreg = [] # y = alpha + beta*x
        
        
    def __str__(self):
        str_self = 'Regresión Lineal | ric ' + self.ric\
            + ' | benchmark ' + self.benchmark + '\n'\
            + 'alpha (intercept) ' + str(self.alpha)\
            + ' | beta (slope) ' + str(self.beta) + '\n'\
            + 'p-value ' + str(self.p_value)\
            + ' | Hipótesis nula  ' + str(self.null_hypothesis) + '\n'\
            + 'r-value (correlación) ' + str(self.r_value)\
            + ' | r-squared ' + str(self.r_squared)
        return str_self
        
    
    def load_timeseries(self):
        self.returns_benchmark, self.returns_ric, self.dataframe = synchronise_timeseries(self.benchmark, self.ric)
    
    
    def compute(self):
        # linear regression del ric con respecto al benchmark
        nb_decimals = 4
        slope, intercept, r_value, p_value, std_err = linregress(self.returns_benchmark,self.returns_ric)
        self.beta = np.round(slope, nb_decimals)
        self.alpha = np.round(intercept, nb_decimals)
        self.p_value = np.round(p_value, nb_decimals) 
        self.null_hypothesis = p_value > 0.05 # p_value < 0.05 --> rechazamos la hipotesis nula 
        self.r_value = np.round(r_value, nb_decimals) #  coeficiente de correlacion
        self.r_squared = np.round(r_value**2, nb_decimals) # porcentaje de varianza de y explicada por x
        self.correlation = self.r_value
        self.predictor_linreg = self.alpha + self.beta*self.returns_benchmark
        
        
    def scatterplot(self):
        # scatterplot de rendimientos
        str_title = 'Diagrama de dispersión de rendimientos' + '\n'
        plt.figure()
        plt.title(str_title)
        plt.scatter(self.returns_benchmark,self.returns_ric)
        plt.plot(self.returns_benchmark, self.predictor_linreg, color='green')
        plt.ylabel(self.ric)
        plt.xlabel(self.benchmark)
        plt.show()
        
        
    def plot_normalised(self):
        # plot 2 series de tiempo normalizadas a 100
        timestamps = self.dataframe['date']
        price_benchmark = self.dataframe['price_1']
        price_ric = self.dataframe['price_2'] 
        plt.figure(figsize=(12,5))
        plt.title('Serie de tiempo de precios | normalizado a 100')
        plt.xlabel('Tiempo')
        plt.ylabel('Precios normalizado')
        price_ric = 100 * price_ric / price_ric[0]
        price_benchmark = 100 * price_benchmark / price_benchmark[0]
        plt.plot(timestamps, price_ric, color='blue', label=self.ric)
        plt.plot(timestamps, price_benchmark, color='red', label=self.benchmark)
        plt.legend(loc=0)
        plt.grid()
        plt.show()
        
        
    def plot_dual_axes(self):
        plt.figure(figsize=(12,5))
        plt.title('Serie de tiempo de precios')
        plt.xlabel('Tiempo')
        plt.ylabel('Precios')
        ax = plt.gca()
        ax1 = self.dataframe.plot(kind='line', x='date', y='price_1', ax=ax, grid=True,
                                  color='blue', label=self.benchmark)
        ax2 = self.dataframe.plot(kind='line', x='date', y='price_2', ax=ax, grid=True,
                                  color='red', secondary_y=True, label=self.ric)
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.show()
        
        
            
    

    
    
###______________________________________________ Cobertura de portafolios ____________________________________________________

class portfolio_manager:
    
    def __init__(self, rics):
        self.rics = rics
        self.nb_decimals = 3
        self.covariance_matrix = [] # annualised
        self.correlation_matrix = [] # annualised
        self.returns = [] # annualised
        self.volatilities = [] # annualised
        
        
    def compute_covariance_matrix(self, bool_print=False):
        # calculamos la matriz de varianza-covarianza por covarianzas por pares
        scale = 252 # anualizado
        size = len(self.rics)
        mtx_covar = np.zeros([size,size])
        mtx_correl = np.zeros([size,size])
        vec_returns = np.zeros([size,1])
        vec_volatilities = np.zeros([size,1])
        returns = []
        for i in range(size):
            ric1 = self.rics[i]
            temp_ret = []
            for j in range(i+1):
                ric2 = self.rics[j]
                ret1, ret2, t = synchronise_timeseries(ric1, ric2)
                returns = [ret1, ret2]
                # covarianzas
                temp_mtx = np.cov(returns)
                temp_covar = scale*temp_mtx[0][1]
                temp_covar = np.round(temp_covar,self.nb_decimals)
                mtx_covar[i][j] = temp_covar
                mtx_covar[j][i] = temp_covar
                # correlaciones
                temp_mtx = np.corrcoef(returns)
                temp_correl = temp_mtx[0][1]
                temp_correl = np.round(temp_correl,self.nb_decimals)
                mtx_correl[i][j] = temp_correl
                mtx_correl[j][i] = temp_correl
                if j == 0:
                    temp_ret = ret1
            # rendimientos
            temp_mean = np.round(scale*np.mean(temp_ret), self.nb_decimals)
            vec_returns[i] = temp_mean
            # volatilidades
            temp_volatility = np.round(np.sqrt(scale)*np.std(temp_ret), self.nb_decimals)
            vec_volatilities[i] = temp_volatility
            
        self.covariance_matrix = mtx_covar
        self.correlation_matrix = mtx_correl
        self.returns = vec_returns
        self.volatilities = vec_volatilities
        
        if bool_print:
            print('-----')
            print('Detalles del Portfolio :')
            print('Securities:')
            print(self.rics)
            print('Rendimientos (anualizados):')
            print(self.returns)
            print('Volatilidades (anualizados):')
            print(self.volatilities)
            print('Matriz de Varianzas-Covarianzas (anualizada):')
            print(self.covariance_matrix)
            print('Matriz de Correlaciones:')
            print(self.correlation_matrix)
            
            
    def compute_portfolio(self, portfolio_type, notional, target_return=None):
        
        size = len(self.rics)
        port_item = portfolio_item(self.rics, notional)
        
        if portfolio_type == 'Varianza-Min':
            port_min_variance, variance_explained = compute_portfolio_min_variance(self.covariance_matrix, notional)
            port_item.type = portfolio_type
            port_item.weights = port_min_variance
            port_item.variance_explained = variance_explained
            
        elif portfolio_type == 'PCA':
            port_pca, variance_explained = compute_portfolio_pca(self.covariance_matrix, notional)
            port_item.type = 'PCA'
            port_item.weights = port_pca
            port_item.variance_explained = variance_explained
            
        elif portfolio_type == 'Long-Only':
            port_long_only = compute_portfolio_long_only(size, notional, self.covariance_matrix)
            port_item.type = 'Long-Only'
            port_item.weights = port_long_only
            
        elif portfolio_type == 'Markowitz':
            if target_return == None:
                target_return = np.mean(self.returns) #analizado
            port_markowitz = compute_portfolio_markowitz(size, notional, self.covariance_matrix,self.returns, target_return)
            port_item.type = 'Markowitz | Rendimiento objetivo ' + str(target_return) 
            port_item.weights = port_markowitz
            port_item.target_return = target_return
            
        else:
            size = len(self.rics)
            port_equi = compute_portfolio_equi_weight(size, notional)
            port_item.type = 'Equi-Ponderado'
            port_item.weights = port_equi
        
        port_item.delta = sum(port_item.weights)
        port_item.pnl_annual = np.dot(port_item.weights.T,self.returns).item()
        port_item.return_annual = port_item.pnl_annual / notional
        port_item.volatility_annual = compute_portfolio_volatility(self.covariance_matrix, port_item.weights)
        if port_item.volatility_annual > 0.0:
            port_item.sharpe_annual =  port_item.return_annual / port_item.volatility_annual
            
        return port_item
            
            
class portfolio_item():
    
    def __init__(self, rics, notional):
        self.rics = rics
        self.notional = notional
        self.type = ''
        self.weights = []
        self.delta = 0.0
        self.pnl_annual = 0.0
        self.return_annual = 0.0
        self.volatility_annual = 0.0
        self.sharpe_annual = 0.0
        self.variance_explained = None
        self.target_return = None


    def summary(self):
        print('-----')
        print('Tipo de Portafolio: ' + self.type)
        print('Rics:')
        print(self.rics)
        print('Pesos:')
        print(self.weights)
        if not self.variance_explained == None:
            print('Varianza explicada: ' + str(self.variance_explained))
        print('Notional (mnUSD): ' + str(self.notional))
        print('Delta (mnUSD): ' + str(self.delta))
        print('PnL anual (mnUSD): ' + str(self.pnl_annual))
        print('Rendimiento anual (mnUSD): ' + str(self.return_annual))
        if not self.target_return == None:
            print('Rendimiento objetivo: ' + str(self.target_return))
        print('Volatilidad anual (mnUSD): ' + str(self.volatility_annual))
        print('Sharpe ratio anual: ' + str(self.sharpe_annual))
     
 



def portafoliosCovarianzas(rics):
    rics = list(rics)
    
    port_mgr = portfolio_manager(rics)
    port_mgr.compute_covariance_matrix(bool_print=False)
    cov_matrix = port_mgr.covariance_matrix
    
    df = pd.DataFrame(cov_matrix, index =rics, columns=rics) 
    
    return df

def portafoliosCorrelaciones(rics):
    rics = list(rics)
    
    port_mgr = portfolio_manager(rics)
    port_mgr.compute_covariance_matrix(bool_print=False)
    corr_matrix = port_mgr.correlation_matrix
    
    df = pd.DataFrame(corr_matrix, index =rics, columns=rics) 
    
    return df

def portafoliosPesos(rics,notional):
    rics = list(rics)
    
    port_mgr = portfolio_manager(rics)
    port_mgr.compute_covariance_matrix(bool_print=False)
    
    #label1 = 'Markowitz' # si no ponemos el hiperparametro extra, calcula el promedio
    #port1 = port_mgr.compute_portfolio('Markowitz', notional,tr_markowitz)
    # label1 = 'min-variance'
    # port1 = port_mgr.compute_portfolio(label1, notional)
    #x1 = port1.volatility_annual
    #y1 = port1.return_annual
    #
    label2 = 'Long-Only'
    port2 = port_mgr.compute_portfolio(label2, notional)
    #
    label3 = 'Equi-Ponderado'
    port3 = port_mgr.compute_portfolio(label3, notional)
    #
    label4 = 'PCA' 
    port4 = port_mgr.compute_portfolio(label4, notional)

    label5 = 'Varianza-Min'
    port5 = port_mgr.compute_portfolio(label5, notional)
    
    label6 = 'Markowitz-Promedio' # si no ponemos el hiperparametro extra, calcula el promedio
    port6 = port_mgr.compute_portfolio('Markowitz', notional)
 
    weights = [port6.weights, port2.weights, port3.weights, port4.weights, port5.weights]
    
    estrategias = ['Markowitz-Promedio','Long-Only','Equi-Ponderado','PCA','Varianza Mínima']
                      
    df = pd.DataFrame(weights, index =estrategias, columns=rics) 
    
    return df
    
#def portafolios(rics,tr_markowitz,notional):

def portafolios(rics,notional):
    rics = list(rics)
    
    port_mgr = portfolio_manager(rics)
    port_mgr.compute_covariance_matrix(bool_print=False)
    
    #label1 = 'Markowitz' # si no ponemos el hiperparametro extra, calcula el promedio
    #port1 = port_mgr.compute_portfolio('Markowitz', notional,tr_markowitz)
    # label1 = 'min-variance'
    # port1 = port_mgr.compute_portfolio(label1, notional)
    #x1 = port1.volatility_annual
    #y1 = port1.return_annual
    #
    label2 = 'Long-Only'
    port2 = port_mgr.compute_portfolio(label2, notional)
    x2 = port2.volatility_annual
    y2 = port2.return_annual
    #
    label3 = 'Equi-Ponderado'
    port3 = port_mgr.compute_portfolio(label3, notional)
    x3 = port3.volatility_annual
    y3 = port3.return_annual
    #
    label4 = 'PCA' 
    port4 = port_mgr.compute_portfolio(label4, notional)
    x4 = port4.volatility_annual
    y4 = port4.return_annual

    label5 = 'Varianza-Min'
    port5 = port_mgr.compute_portfolio(label5, notional)
    x5 = port5.volatility_annual
    y5 = port5.return_annual
    
    label6 = 'Markowitz-Promedio' # si no ponemos el hiperparametro extra, calcula el promedio
    port6 = port_mgr.compute_portfolio('Markowitz', notional)
    x6 = port6.volatility_annual
    y6 = port6.return_annual
    
    #data = {'Markowitz':[port1.weights, port1.pnl_annual, port1.return_annual, 
    #                     port1.target_return, port1.volatility_annual, port1.sharpe_annual],
    #        'Markowitz-Promedio':[port6.weights, port6.pnl_annual, port6.return_annual, 
    #                     port6.target_return, port6.volatility_annual, port6.sharpe_annual],
    #        'Long-Only':[port2.weights, port2.pnl_annual, port2.return_annual, 
    #                     port2.target_return, port2.volatility_annual, port2.sharpe_annual],
    #        'Equi-Ponderado':[port3.weights, port3.pnl_annual, port3.return_annual, 
    #                     port3.target_return, port3.volatility_annual, port3.sharpe_annual],
    #        'PCA':[port4.weights, port4.pnl_annual, port4.return_annual, 
    #                     port4.target_return, port4.volatility_annual, port4.sharpe_annual],
    #        'Varianza Mínima':[port5.weights, port5.pnl_annual, port5.return_annual, 
    #                     port5.target_return, port5.volatility_annual, port5.sharpe_annual]} 
    
    data = {'Markowitz-Promedio':[port6.pnl_annual, port6.return_annual, 
                         port6.target_return, port6.volatility_annual, port6.sharpe_annual],
            'Long-Only':[port2.pnl_annual, port2.return_annual, 
                         port2.target_return, port2.volatility_annual, port2.sharpe_annual],
            'Equi-Ponderado':[port3.pnl_annual, port3.return_annual, 
                         port3.target_return, port3.volatility_annual, port3.sharpe_annual],
            'PCA':[port4.pnl_annual, port4.return_annual, 
                         port4.target_return, port4.volatility_annual, port4.sharpe_annual],
            'Varianza Mínima':[port5.pnl_annual, port5.return_annual, 
                         port5.target_return, port5.volatility_annual, port5.sharpe_annual]} 

    df = pd.DataFrame(data, index =['PnL anual (mnUSD)','Rendimiento anual (mnUSD)',
                                    'Rendimiento objetivo','Volatilidad anual (mnUSD)','Sharpe ratio anual']) 
    
    return df


#def graficaPortafolios(rics,tr_markowitz,notional):
def graficaPortafolios(rics,notional):
    rics = list(rics)
    
    port_mgr = portfolio_manager(rics)
    port_mgr.compute_covariance_matrix(bool_print=False)
    
    # calculamos vectores de rendimientos y volatilidades para los portafolios de Markowitz
    min_returns = np.min(port_mgr.returns)
    max_returns = np.max(port_mgr.returns)
    returns = min_returns + np.linspace(0.1,0.9,100) * (max_returns-min_returns)
    volatilities = np.zeros([len(returns),1])
    counter = 0
    for target_return in returns:
        port_markowitz = port_mgr.compute_portfolio('Markowitz', notional, target_return)
        volatilities[counter] = port_markowitz.volatility_annual
        counter += 1
    
    #label1 = 'Markowitz' # si no ponemos el hiperparametro extra, calcula el promedio
    #port1 = port_mgr.compute_portfolio('Markowitz', notional,tr_markowitz)
    # label1 = 'min-variance'
    # port1 = port_mgr.compute_portfolio(label1, notional)
    #x1 = port1.volatility_annual
    #y1 = port1.return_annual
    #
    label2 = 'Long-Only'
    port2 = port_mgr.compute_portfolio(label2, notional)
    x2 = port2.volatility_annual
    y2 = port2.return_annual
    #
    label3 = 'Equi-Ponderado'
    port3 = port_mgr.compute_portfolio(label3, notional)
    x3 = port3.volatility_annual
    y3 = port3.return_annual
    #
    label4 = 'PCA' 
    port4 = port_mgr.compute_portfolio(label4, notional)
    x4 = port4.volatility_annual
    y4 = port4.return_annual

    label5 = 'Varianza-Min'
    port5 = port_mgr.compute_portfolio(label5, notional)
    x5 = port5.volatility_annual
    y5 = port5.return_annual
    
    label6 = 'Markowitz-Promedio' # si no ponemos el hiperparametro extra, calcula el promedio
    port6 = port_mgr.compute_portfolio('Markowitz', notional)
    # label1 = 'min-variance'
    # port1 = port_mgr.compute_portfolio(label1, notional)
    x6 = port6.volatility_annual
    y6 = port6.return_annual
    #
    
    figure = plt.figure(figsize=(12,8))
    plt.scatter(volatilities,returns)
    #plt.plot(x1, y1, "^k", label=label1, markersize=12) # black triangle
    plt.plot(x2, y2, "sm", label=label2, markersize=12) # magenta square
    plt.plot(x3, y3, "db", label=label3, markersize=12) # blue diamond
    plt.plot(x4, y4, "*g", label=label4, markersize=12) # green star
    plt.plot(x5, y5, "hr", label=label5, markersize=12) # red hexagon
    plt.plot(x6, y6, "py", label=label6, markersize=12) # yellow pentagon
    plt.ylabel('Rendimiento del portafolio', fontsize=14)
    plt.xlabel('Volatilidad del portafolio', fontsize=14)
    plt.grid()
    plt.legend(loc="best")

    return (figure)
    
##_____________________________ StreamlitFRONT ________________________________________
    
logo = Image.open("logoUNAM.png")
logo2 = Image.open("IIMAS.png")
dinero = Image.open("portafolio2.png")

# Formato
#st.image([logo,logo2],use_column_width=False, width=100)
#st.image(logo,use_column_width=False, width=100)
#st.image(logo2,use_column_width=False, width=100)
#st.sidebar.image(dinero, width=300)
st.sidebar.image([logo,logo2],use_column_width=False, width=100)
menu = st.sidebar.radio('Menu',('Inicio','Prueba de Jarque-Bera','Modelo CAPM', 'Cobertura de Portafolios','Referencias','Información'))

if menu == 'Inicio':
    st.title("Finanzas Cuantitativas")
    st.image(dinero, width=700)
    st.write('En cierto sentido, la tecnología per se no es nada especial para las instituciones financieras (en comparación, por ejemplo, con las empresas industriales) o para la función financiera (en comparación con otras funciones corporativas, como logística). Sin embargo, en los últimos años, impulsados por la innovación y también la regulación, los bancos y otras instituciones financieras como los fondos de cobertura han evolucionado cada vez más en empresas de tecnología en lugar de ser solo intermediarios financieros. La tecnología se ha convertido en un activo importante para casi cualquier institución financiera del mundo, y tiene el potencial de generar ventajas competitivas.')

    st.write('Los bancos y las instituciones financieras forman la industria que más gasta en tecnología anualmente. Los grandes bancos multinacionales hoy en día generalmente emplean a miles de desarrolladores que mantienen los sistemas existentes y construyen otros nuevos. Se estima que el gasto total en Tecnologías de la Información en servicios financieros a nivel mundial fue de \$500 mil millones.')

    st.write('Hay una disciplina que ha experimentado un fuerte aumento de importancia en la industria financiera: análisis financiero y de datos. Este fenómeno tiene una estrecha relación con la percepción de que las velocidades, frecuencias y volúmenes de datos aumentan a un ritmo rápido en la industria. De hecho, la analítica en tiempo real puede considerarse la respuesta de la industria a esta tendencia.')
    
    st.write('Por ello dentro de este proyecto, realizaremos un análisisde datos de Finanzas Cuantitativas mediante el lenguaje de programación Python')
    
    
if menu == 'Prueba de Jarque-Bera':
    st.subheader("Prueba de Jarque-Bera")
    st.write("La prueba de Jarque-Bera es realizada a un conjunto de datos con el objetivo de determinar \
    si se trata o no de una distribución normal. El estadístico se distribuye asintóticamente como una distribución chicuadrado con 2 grados de libertad.")
    st.write("Se puede utilizar en modelos de regresión para probar la hipótesis de normalidad de los residuos. Para lo que se usan los estimados obtenidos por mínimos cuadrados. ")
    st.write("En caso de que el valor de Jaquer-Bera sea menor a 6, será considerada una distribución normal")
    st.write("Está dada por la siguiente fórmula ")
    r'''
    $$JB = \frac{n}{6}\left(S^2+\frac{1}{4}(K-3)^2\right)$$

    Donde:

    $$S = \frac{\hat{\mu}_3}{\hat{\sigma}^3} = \frac{\frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^3}{(\frac{1}{n} \sum_{i=1}^n(x_i - \bar{x})^2)^3/2}$$

    $$K = \frac{\hat{\mu}_4}{\hat{\sigma}^4} = \frac{\frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^4}{(\frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2)^2}$$

    '''
    st.write("Siendo:")
    st.write("K (Kurtosis): Medida que determina qué tan achatada se encuentra nuestra curva normal. Entre mayor curtosis los datos estarán más cerca a su media (Mayor pico), mientras que cuanto menor sea el valor, los datos presentan un mayor alejamiento de su media (Curva achatada).")
    st.write("S (Skewness): Coeficiente de asimetría. Indica hacia dónde se encuentran sesgados los datos. Un valor de 0, indicará una perfecta simetría y un nulo desbalance. Coeficientes mayores a 1 o menores a -1 indicarán una distribución altamente sesgada. Coeficientes entre [-1,-0.5] o [1,0.5] nos hablan de sesgo moderado y finalmente valores entrre -0.5 y 0.5 habland de un sesgo aproximado")

    #Nuestras posibles bases
    display = ('AAL.L', 'MXNUSD=X', '^FCHI', 'SOGN.PA', 'TOTF.PA', 'RIO.L', 'DBK.DE', 'VWS.CO', 'MXN=X', 'SAN.MC', 
               'EURUSD=X', 'CBKG.DE', 'DBKGn.DE', 'REP.MC', '^VIX', 'BNPP.PA', '^NASDAQ', '^GDAXI', 'EQNR.OL', 
               'CHFUSD=X', 'BP.L', 'EDPR.LS', '^S&P500', 'RDSa.AS', 'EDF.PA', 'GBPUSD=X', 'MT.AS', '^STOXX', 
               'EONGn.DE', 'GLEN.L', 'KBC.BR', 'SGREN.MC', 'ISP.MI', 'ANTO.L', '^STOXX50E', 'RWEG.DE', 
               'INGA.AS', '^IXIC', 'RDSa.L', 'EDP.LS', 'CRDI.MI', 'BBVA.MC')

    #Nuestro "diccionario"
    options = list(range(len(display)))

    #La lectura del valor
    value = st.selectbox("Selecciona un conjunto de datos", options, format_func=lambda x: display[x])

    #Lectura de datos
    path = "Datos/"+display[value]+".csv"
    table_raw = pd.read_csv(path)

    #Creación del dataframe
    t = pd.DataFrame()
    #Convertimos date to datetime
    t['Fecha'] = pd.to_datetime(table_raw['Date'], dayfirst=True)
    #Extraemos el cierre
    t['Cierre'] = table_raw['Close']
    #Ordenamos nuestros valores por día
    t.sort_values(by='Fecha', ascending=True)
    # Obtenemos el cierre anterior
    t['Cierre_Anterior'] = t['Cierre'].shift(1)
    # Obtenemos el retorno de cierre
    t['Retorno_al_cierre'] = t['Cierre']/t['Cierre_Anterior'] - 1
    #Eliminamos nuestro valor que sobra (el primero)
    t = t.dropna()
    #REseteamos el índice
    t = t.reset_index(drop=True)

    title = 'Precio al cierre por día ' +display[value]


    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['x'], empty='none')


    C = alt.Chart(t).mark_area(
        line={'color':'darkgreen'},
        color=alt.Gradient(
            gradient='linear',
            stops=[alt.GradientStop(color='white', offset=0),
                   alt.GradientStop(color='darkgreen', offset=1)],
        )
    ).encode(
        alt.X('Fecha:T', title="Fecha"),
        alt.Y('Cierre:Q', title="Valor"),
        tooltip = [alt.Tooltip('Fecha:N'),
                   alt.Tooltip('Cierre:N'),
                   alt.Tooltip('Retorno_al_cierre:N')
                  ]
    ).properties(
        width=650, 
        height=300,
        title=title,
    ).add_selection(
        nearest
    )

    st.header("Visualizando el comportamiento del valor al cierre")
    st.write(C)

    st.header("Nuestros datos")
    st.dataframe(t)

    status_text = st.sidebar.empty()

    fig = ff.create_distplot([t['Retorno_al_cierre'].values],['x'], bin_size=[.001],colors=['#835AF1'])
    fig.update_layout(title_text='HIstograma para el retorno al cierre\n'+display[value],width=800, height=600,showlegend=False
    )

    st.header("Visualizando la distribución")
    st.write("A continuación, presentamos un histograma con los datos en cuanto al retorno del cierre, con el objetivo de ayudar a visualizar la distribución a analizar. \n Como podemos observar, resultaría sencillo equivocarse y asumir que la distribución presente representa una normal. Es por ello que procedemos a verificarlo a través de métodos matemáticos.")

    st.plotly_chart(fig, use_container_width=True)

    ### Aquí comienza lo de Jarque-Bera

    # input for Jarque-Bera test
    x = t['Retorno_al_cierre'].values
    x_str = 'Retorno al cierre ' + display[value]
    x_size = len(x) # Tamaño del retorno




    # compute "risk metrics"
    x_mean = stats.tmean(x)
    x_std = stats.tstd(x) # volatility
    x_skew = stats.skew(x)
    x_kurt = stats.kurtosis(x) # excess kurtosis
    x_sharpe = x_mean / x_std * np.sqrt(252) # annualised
    x_var_95 = stats.percentileofscore(x,5)
    x_cvar_95 = stats.tmean(x[x <= x_var_95])
    jb = x_size/6*(x_skew**2 + 1/4*x_kurt**2)
    p_value = 1 - stats.chi2.cdf(jb, df=2)
    is_normal = (p_value > 0.05) # equivalently jb < 6


    st.header("Analizando resultados")

    round_digits = 4
    data = [np.round(x_mean,round_digits), np.round(x_skew,round_digits), np.round(x_kurt,round_digits),
           np.round(x_sharpe,round_digits),np.round(x_var_95,round_digits),
           np.round(x_cvar_95,round_digits), np.round(jb,round_digits), np.round(p_value,round_digits)]
    cols = ["Media", "Skewness", "Kurtosis","Sharpe ratio", "Valor en Riesgo (VaR) 95%","Déficit esperado (CVaR) 95%","Jarque-Bera","p"]

    aux = pd.DataFrame(columns=cols)
    aux.loc[0]=data
    st.table(aux)

    st.header("Conclusión")

    if is_normal:

        st.header("Representa una distribución normal")
    else:
        st.header("No Representa una distribución normal")

        
        
###______________________Front CAPM_________________________

if menu == 'Modelo CAPM':
    st.subheader("Modelo CAPM")
    st.write('El Modelo de valoración de activos financieros permite estimar la rentabilidad esperada en función del riesgo sistemático, este modelo afirma que la ganancia de una inversión será mayor en medida que el riesgo de dicha acción sea mayor.')

    
    ric = ('SAN.MC','MT.AS','SAN.MC','BBVA.MC','REP.MC','VWS.CO','EQNR.OL','MXNUSD=X','^VIX','GBPUSD=X','CHFUSD=X')
    benchmark = ('^STOXX50E','^STOXX50E','^STOXX','^S&P500','^NASDAQ','^FCHI','^GDAXI','EURUSD=X')

    options = list(range(len(ric)))
    value = st.selectbox("Selecciona un conjunto de datos para ric", options, format_func=lambda x: ric[x])
    
    options2 = list(range(len(benchmark)))
    value2 = st.selectbox("Selecciona un conjunto de datos para benchmark", options2, format_func=lambda x: benchmark[x])
    
    #Lectura de datos
    path = "Datos/"+ric[value]+".csv"
    table_raw = pd.read_csv(path)
    st.write('RIC')
    st.dataframe(table_raw)
    
    path2 = "Datos/"+benchmark[value2]+".csv"
    table_raw_mad = pd.read_csv(path2)
    st.write('Benchmark')
    st.dataframe(table_raw_mad)
    st.subheader('Datos')
    
      
    st.subheader('Viendo nuestros datos como Series de tiempo')
    #Creación del dataframe
    t = pd.DataFrame()
    #Convertimos date to datetime
    t['Fecha'] = pd.to_datetime(table_raw['Date'], dayfirst=True)
    #Extraemos el cierre
    t['Cierre'] = table_raw['Close']
    #Ordenamos nuestros valores por día
    t.sort_values(by='Fecha', ascending=True)
    # Obtenemos el cierre anterior
    t['Cierre_Anterior'] = t['Cierre'].shift(1)
    # Obtenemos el retorno de cierre
    t['Retorno_al_cierre'] = t['Cierre']/t['Cierre_Anterior'] - 1
    #Eliminamos nuestro valor que sobra (el primero)
    t = t.dropna()
    #REseteamos el índice
    t = t.reset_index(drop=True)
    st.write("Benchmark (Time Series)")
    st.dataframe(t)
    
    #Creación del dataframe
    t = pd.DataFrame()
    #Convertimos date to datetime
    t['Fecha'] = pd.to_datetime(table_raw_mad['Date'], dayfirst=True)
    #Extraemos el cierre
    t['Cierre'] = table_raw_mad['Close']
    #Ordenamos nuestros valores por día
    t.sort_values(by='Fecha', ascending=True)
    # Obtenemos el cierre anterior
    t['Cierre_Anterior'] = t['Cierre'].shift(1)
    # Obtenemos el retorno de cierre
    t['Retorno_al_cierre'] = t['Cierre']/t['Cierre_Anterior'] - 1
    #Eliminamos nuestro valor que sobra (el primero)
    t = t.dropna()
    #REseteamos el índice
    t = t.reset_index(drop=True)
    st.write("RIC (Time Series)")
    st.dataframe(t)
    
    st.subheader('Series de tiempo sincronizadas')
    x,y, t_s = synchronise_timeseries('^STOXX50E', 'SAN.MC')
    st.dataframe(t_s)
    
    ric = str(ric[value])
    benchmark = str(benchmark[value2])
    capm = capm_manager(benchmark, ric)
    capm.load_timeseries()
    capm.compute()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write('Series de tiempo (Precios Normalizados)')
    st.pyplot(capm.plot_normalised())
    st.write('Series de tiempo (Precios)')
    st.pyplot(capm.plot_dual_axes())
    
    
    st.subheader('Aplicando regresión')
    st.pyplot(capm.scatterplot())
    st.subheader('Resultados')
    st.write(capm)



    
    
    
    
    
if menu == 'Cobertura de Portafolios':
    
    st.subheader("Cobertura de Portafolios")

    #rics = st.selectbox("Conjuntos de acciones",  [['^S&P500','^VIX'],
    #                                                   ['SAN.MC','BBVA.MC','SOGN.PA','BNPP.PA','INGA.AS','KBC.BR'], 
    #                                                   ['MXNUSD=X','EURUSD=X','GBPUSD=X','CHFUSD=X'],
    #                                                   ['SAN.MC','BBVA.MC','SOGN.PA','BNPP.PA','INGA.AS','KBC.BR','CRDI.MI',
    #                                                    'ISP.MI','DBKGn.DE','CBKG.DE'],
    #                                                   ['SGREN.MC','VWS.CO','TOTF.PA','REP.MC','BP.L','RDSa.AS','RDSa.L'],
    #                                                   ['SGREN.MC','VWS.CO'],
    #                                                   ['TOTF.PA','REP.MC','BP.L','RDSa.AS','RDSa.L'],
    #                                                   ['AAL.L','ANTO.L','GLEN.L','MT.AS','RIO.L']]
    #




    notional = st.number_input("Monto nocional", value=10)
    notional = float(notional)
    st.write("El monto nocional ingresado es ", np.round(notional,0))

    #tr_markowitz = st.number_input('Rendimiento objetivo para el portafolio de Markowitz: ', min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    #tr_markowitz = float(tr_markowitz)
    #st.write("El rendimiento objetivo ingresado para Markowitz es ", np.round(tr_markowitz,2))

    rics = st.multiselect('Seleccione el conjunto de acciones', 
                          ['^S&P500','^VIX','MXNUSD=X','EURUSD=X','GBPUSD=X','CHFUSD=X','SAN.MC',
                           'BBVA.MC','SOGN.PA','BNPP.PA','INGA.AS','KBC.BR','CRDI.MI','ISP.MI','DBKGn.DE',
                           'CBKG.DE','SGREN.MC','VWS.CO','TOTF.PA','REP.MC','BP.L','RDSa.AS','RDSa.L',
                           'AAL.L','ANTO.L','GLEN.L','MT.AS','RIO.L'], 
                           default=['SAN.MC', 'BBVA.MC','SOGN.PA','BNPP.PA','INGA.AS','KBC.BR',
                                    'CRDI.MI','ISP.MI','DBKGn.DE','CBKG.DE'])   

    #rics = st.selectbox("Conjuntos de acciones",  [['^S&P500','^VIX'],
    #                                                   ['SAN.MC','BBVA.MC','SOGN.PA','BNPP.PA','INGA.AS','KBC.BR'], 
    #                                                   ['MXNUSD=X','EURUSD=X','GBPUSD=X','CHFUSD=X'],
    #                                                   ['SAN.MC','BBVA.MC','SOGN.PA','BNPP.PA','INGA.AS','KBC.BR','CRDI.MI',
    #                                                    'ISP.MI','DBKGn.DE','CBKG.DE'],
    #                                                   ['SGREN.MC','VWS.CO','TOTF.PA','REP.MC','BP.L','RDSa.AS','RDSa.L'],
    #                                                   ['SGREN.MC','VWS.CO'],
    #                                                   ['TOTF.PA','REP.MC','BP.L','RDSa.AS','RDSa.L'],
    #                                                   ['AAL.L','ANTO.L','GLEN.L','MT.AS','RIO.L']])


    st.subheader("Matriz de Varianzas-Covarianzas de los rics del portafolio")

    st.dataframe(portafoliosCovarianzas(tuple(rics)))

    st.subheader("Matriz de Correlaciones de los rics del portafolio")

    st.dataframe(portafoliosCorrelaciones(tuple(rics)))

    st.subheader("Tabla de Pesos de los rics del portafolio")

    st.dataframe(portafoliosPesos(tuple(rics),notional))

    st.subheader("Información de los portafolios")

    #st.dataframe(portafolios(tuple(rics), tr_markowitz, notional))
    st.dataframe(portafolios(tuple(rics), notional))
    
    
    st.subheader("Gráfica de la Frontera Eficiente de los rics del portafolio")

    #st.pyplot(graficaPortafolios(rics,tr_markowitz,notional))
    st.pyplot(graficaPortafolios(rics, notional))

 



###________ Referencias__________ 

if menu == 'Referencias':
    st.subheader('Referencias')
    text_string_variable = 'Modelo de valoración de activos financieros'
    url_string_variable = 'https://economipedia.com/definiciones/modelo-valoracion-activos-financieros-capm.html'
    link = f'[{text_string_variable}]({url_string_variable})'
    st.markdown(link, unsafe_allow_html=True)
    
    text_string_variable = 'Modelo CAPM para calcular el precio de los activos'
    url_string_variable = 'https://www.rankia.co/blog/como-comenzar-invertir-bolsa/3324781-modelo-capm-para-calcular-precio-activos'
    link = f'[{text_string_variable}]({url_string_variable})'
    st.markdown(link, unsafe_allow_html=True)
    
    text_string_variable = 'Modelo de fijación de precios de activos de capital (CAPM)'
    url_string_variable = 'https://qsstudy.com/business-studies/capital-asset-pricing-model-capm'
    link = f'[{text_string_variable}]({url_string_variable})'
    st.markdown(link, unsafe_allow_html=True)
    
    text_string_variable = 'Estrategias de cobertura'
    url_string_variable = 'https://www.eleconomista.com.mx/mercados/Que-son-y-como-funcionan-las-estrategias-de-cobertura-20140725-0026.html'
    link = f'[{text_string_variable}]({url_string_variable})'
    st.markdown(link, unsafe_allow_html=True)
    
    text_string_variable = 'Actividad de Mercados'
    url_string_variable = 'https://www.nasdaq.com/'
    link = f'[{text_string_variable}]({url_string_variable})'
    st.markdown(link, unsafe_allow_html=True)
    
    text_string_variable = 'Test de Jarque-Bera'
    url_string_variable = 'https://es.wikipedia.org/wiki/Test_de_Jarque-Bera'
    link = f'[{text_string_variable}]({url_string_variable})'
    st.markdown(link, unsafe_allow_html=True)
    
    text_string_variable = 'Sharpe Ratio'
    url_string_variable = 'https://www.investopedia.com/articles/07/sharpe_ratio.asp'
    link = f'[{text_string_variable}]({url_string_variable})'
    st.markdown(link, unsafe_allow_html=True)
    
    
    
    
### _______ Información ____________________________
if menu == 'Información':
    st.subheader("Universidad Nacional Autónoma de México")
    st.subheader("Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas")
    st.subheader("Introducción a las Finanzas y la Empresa")
    st.subheader("Aplicación creada por: ")
    st.write('Martiñón Luna Jonathan José')
    st.write('Ortega Ibarra Jaimes Jesús')
    st.write('Tapia López José de Jesús')
    st.write("Enero del 2021")




