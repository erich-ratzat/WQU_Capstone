# -*- coding: utf-8 -*-
"""
(WQU 690 Capstone (C18-S4T5)
Capstone Project:
Does Piotroski F-Score work in Emerging Markets? The Brazilian market analysis
Author: Erich Leonardo Ratzat
"""


# ################################# ABSTRACT #################################

# The objective of this paper is to test Piotroski (2000) model in the 
# Brazilian Market for the period 2005-2018. In addition, apply the momentum 
# strategy after the Piotroski portfolio selection to identify if there is any
# improvement in returns and risk.
# ############################################################################


# Import the following libraries:
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

# Select the source files 

# Below are the data source containing the raw data for all 9 indicators:
# Please copy all the excel files to the same working directory

df_roa = '1-ROA - anual.xlsx' # roa > 0 then 1, else 0
#roa_t > roa_t-1 then 1, else 0
df_cfo = '2-CaixaeOp_v2.xlsx' # cf > 0 then 1, else 0
df_accrual = '3-Accrual.xlsx' # accrual_inverted > 0 then 1, else 0
df_liquidity = '5-Liquidez Corrente - anual.xlsx' # liq diff > 0 then 1, else 0
df_debt = '6-Endividamento - anual.xlsx' # debt diff > 0 then 0, else 1 *inverted*
df_stock_issue = '7-Qtde_Acoes - anual.xlsx' # issue diff != 0 then 0, else 1 *inverted*
df_margin = '8-Margem_Bruta - anual.xlsx' # margin diff > 0 then 1, else 0 
df_turnover = '9-Giro_Ativo - anual.xlsx' # turnover diff > 0 then 1, else 0
df_pb = '10-PB - anual.xlsx' # Price to Book 
pb = pd.read_excel(df_pb, index_col='Date')
close = pd.read_excel('monthly_close_clean_v1.xlsx', index_col='Date')
# calculate Return for the last 6 months
ret6 = close.pct_change(6)

# Below are the main functions to calculate Piotroski F Score and performance
def clean_data(data):
    '''
    clean excel files to work with only
    float and not strings
    data: it is the DataFrame from the Excel file
    '''
    
    df = pd.read_excel(data, index_col='Date')
    df = df.replace('-','')
    df = df.drop('conso',1)
    df = df.apply(pd.to_numeric) # convert all columns of DataFrame

    global label_columns2
    global label_index2
    label_columns2 = df.columns
    label_index2 = df.index

    return df


def array_to_df(data):
    '''
    transform np.array into DataFrame and label
    the columns and the index correctly
    data: it is the DataFrame from the Excel file
    '''
    df = pd.DataFrame(data, index=range(data.shape[0]),
                          columns=range(data.shape[1]))
    df.columns = label_columns2
    df.index = label_index2
    return df


def condition_cal(data, tipo, inverted=False):
    '''
    tipo means what the kind of condition. For example,
    Piotroski works with 2 types of condition: greater than zero
    or the actual value is greater the the previous value.
    data: it is the DataFrame from the Excel file
    tipo: Here the user must choose between :
          tipo == 'diff' which means greater than previou value or
          tipo == 'greater' which means greater than zero
    inverted: two indicators has inverted logic, if greater than zero
    then receives 0 otherwise receives 1. The standard is False. When
    working with inverted logic please insert "True".
    '''
    if tipo == 'diff':
        if inverted==True:
            condition = np.where(data.diff() < 0, 1, 0)
        elif inverted==False:
            condition = np.where(data.diff() > 0, 1, 0)
            
    elif tipo == 'greater':
        condition = np.where(data > 0, 1, 0)

    elif tipo == 'eq_offer':
        condition = np.where(data.diff() > 0, 0, 1)
        
    return condition


class Output(object):
    def __init__(self, returns_df, date_freq='A'):
        self.returns_df = returns_df if isinstance(
                returns_df, pd.DataFrame) else pd.DataFrame(returns_df)
        self.wealthpaths = self.returns_df.apply(self._calc_wealthpath)
        self._date_freq = str(date_freq).upper()
        if self._date_freq == 'D':
            self._freq = 252
        elif self._date_freq == 'M':
            self._freq = 12
        elif self._date_freq == 'A':
            self._freq = 1

    def _calc_annualized_return(self, series):
        avg_daily_return = series.mean()
        ann_return = avg_daily_return * self._freq
        return ann_return

    def _calc_annualized_std_dev(self, series):
        series_std = series.std()
        ann_std = series_std * (np.sqrt(self._freq))
        return ann_std

    def _calc_sharpe(self, ann_returns, ann_stds):
        sharpe = ann_returns.divide(ann_stds)
        return sharpe

    def _calc_hwm(self, wealthpath):
        hwm = wealthpath.expanding().max()
        return hwm

    def _calc_wealthpath(self, series):
        if series.iloc[0] != 0:
            first_dt = series.index[0]
            set_dt = first_dt - dt.timedelta(days=1)
            series.loc[set_dt] = 0.0
            series = series.sort_index()

        cum_prod = (1.0 + series).cumprod()
        return cum_prod

    def _calc_drawdowns(self, wealthpath):
        hwm = self._calc_hwm(wealthpath)
        drawdowns = wealthpath.divide(hwm).subtract(1.0)
        return drawdowns

    def _calc_lake_ratios(self, hwm, wps):
        lakes = hwm.subtract(wps)
        mountains = hwm.subtract(lakes)
        lake_ratios = lakes.sum() / mountains.sum()
        return lake_ratios

    def _calc_gain_to_pain_ratio(self, series):
        total_return_series = (1.0 + series).cumprod().subtract(1.0)
        total_return = total_return_series.iloc[-1]

        loss_returns_series = self.__get_loss_returns(series).abs()
        if not loss_returns_series.empty:
            total_loss_return_series = (1.0 + loss_returns_series).cumprod().subtract(1.0)
            total_loss_return = total_loss_return_series.iloc[-1]

            gpr = total_return / total_loss_return
        else:
            gpr = np.nan
        return gpr

    def __get_win_returns(self, series):
        win_returns = series[series >= 0.0]
        return win_returns

    def __get_loss_returns(self, series):
        loss_returns = series[series < 0.0]
        return loss_returns

    def _calc_win_rate(self, series):
        win_returns = self.__get_win_returns(series)
        rate = float(len(win_returns)) / float(len(series))
        return rate

    def _calc_loss_rate(self, series):
        loss_returns = self.__get_loss_returns(series)
        rate = float(len(loss_returns)) / float(len(series))
        return rate

    def _calc_avg_win_return(self, series):
        win_returns = self.__get_win_returns(series)
        avg = win_returns.mean()
        return avg

    def _calc_avg_loss_return(self, series):
        loss_returns = self.__get_loss_returns(series)
        avg = loss_returns.mean()
        return avg

    def _calc_winloss_ratio(self, series):
        wins = self.__get_win_returns(series)
        losses = self.__get_loss_returns(series)
        if len(losses) == 0.0:
            wl_ratio = np.nan
        else:
            wl_ratio = len(wins) / len(losses)
        return wl_ratio

    def _calc_expectancy(self, win_rates, avg_win, loss_rates, avg_loss):
        w_win = win_rates.multiply(avg_win)
        w_loss = loss_rates.multiply(avg_loss)
        exp = w_win.subtract(w_loss)
        return exp

    def generate_output(self):
        hwms = self.wealthpaths.apply(self._calc_hwm)
        lake_ratios = self._calc_lake_ratios(hwms, self.wealthpaths)
        lake_ratios.name = "Lake Ratio"

        drawdowns = self.wealthpaths.apply(self._calc_drawdowns)
        max_dds = drawdowns.min()
        max_dds.name = "Max Drawdown"

        ann_returns = self.returns_df.apply(self._calc_annualized_return)
        ann_returns.name = "Annualized Return"

        ann_stds = self.returns_df.apply(self._calc_annualized_std_dev)
        ann_stds.name = "Annualized Std Dev"

        sharpes = self._calc_sharpe(ann_returns, ann_stds)
        sharpes.name = "Sharpe Ratio"

        win_rates = self.returns_df.apply(self._calc_win_rate)
        win_rates.name = "Win Rate"

        loss_rates = self.returns_df.apply(self._calc_loss_rate)
        loss_rates.name = "Loss Rate"

        avg_win_returns = self.returns_df.apply(self._calc_avg_win_return)
        avg_win_returns.name = "Avg Win Return"

        avg_loss_returns = self.returns_df.apply(self._calc_avg_loss_return)
        avg_loss_returns.name = "Avg Loss Return"

        win_loss_ratio = self.returns_df.apply(self._calc_winloss_ratio)
        win_loss_ratio.name = "Win Loss Ratio"

        expectancy = self._calc_expectancy(win_rates, avg_win_returns,
                                           loss_rates, avg_loss_returns)
        expectancy.name = "Trade Expectancy"

        gpr = self.returns_df.apply(self._calc_gain_to_pain_ratio)
        gpr.name = 'Gain to Pain Ratio'

        output_df = pd.concat([lake_ratios, max_dds, ann_returns,
                               ann_stds, sharpes, win_rates,
                               loss_rates, avg_win_returns,
                               avg_loss_returns, expectancy,
                               gpr, ], axis=1).round(4)

        return output_df.T.sort_index()
    

def portfolio_returns():
    '''
    calculates the portfolio returns for Piotroski, Momentum and Bovespa
    for the period 2005 to 2018
    '''
    labels = ['2005-12-31', '2006-12-31', '2007-12-31', '2008-12-31', '2009-12-31', '2010-12-31', '2011-12-31', '2012-12-31', '2013-12-31', '2014-12-31', '2015-12-31', '2016-12-31', '2017-12-31', '2018-12-31']
    piotroski = []
    momentum = []
    ibov = []
    for i in range(2004,2018):
        p = piotroski_return_switch(i, 6, 0, 'long', 3)
        m = piotroski_return_momentum_switch(i, 6, 0, 0, 'long', 3)
        i = ibov_return(close, i)
        piotroski.append(p)
        momentum.append(m)
        ibov.append(i)
        
    df = pd.DataFrame({'Ibovespa':ibov, 'Momentum':momentum, 'Piotroski':piotroski}, index=labels)
    df.index.name = 'Date'
    df.index = pd.to_datetime(labels)
    #df.index = pd.DatetimeIndex(df.index).year
    
    return df


def piotroski_rank(year, score_long, score_short, long_short):
    '''
    calculates Piotroski Rank by year. It shows the stocks
    with its respective F Score above the respective cut off "score"
    year: choose the year to calculate F Score
    score: choose the cut off for F Score - between 0 and 9.
    '''    
    calculation = conso.loc[year]
    calculation = calculation.to_frame().reset_index()
    calculation = calculation.rename(columns = {'index': 'Stock', year: 'F Score'})
    calculation.index.name = str(year)
    if long_short == 'long':
        return calculation[calculation['F Score'] > score_long].sort_values('F Score', ascending=False)
    if long_short == 'long_short':
        long_port = calculation[calculation['F Score'] > score_long]
        short_port = calculation[calculation['F Score'] < score_short]
        long_short_port = pd.concat([long_port, short_port], ignore_index=True)

        return long_short_port


def price_to_book(year, score):
    '''
    calculates the price to book metric. It is the inverted Book Market metric.
    For Piotroski model it should be considered low price to book numbers.
    '''
    pb_calculation = pb.loc[year]
    pb_calculation = pb_calculation.to_frame().reset_index()
    pb_calculation = pb_calculation.rename(columns = {'index': 'Stock', year: 'Price to Book'})
    pb_calculation.index.name = str(year)
    return pb_calculation[pb_calculation['Price to Book'] > score].sort_values('Price to Book', ascending=True)


def find_percentile2(year, quantile):
    '''
    calculates the percentile by year to select high book to market firms - HBM
    here is used the inverse of HBM which is the price to book
    '''
    df = price_to_book(year,0)
    df2 = df['Price to Book'].quantile(quantile)
    return df2


def ibov_return(data, year):
    '''
    calculate the index return
    '''
    ibov_return = close['IBOV']
    if year == 2017:
        year_buy = str(year+1)+'-04-30'
        year_sell = str(year+2)+'-01-31'
        return_year = (ibov_return.loc[year_sell] / ibov_return.loc[year_buy]-1)
        #return_year = return_year.to_frame().reset_index()
        #return_year = return_year.rename(columns = {'index': 'Stock', 0: 'Return'})
    
    else:
        year_buy = str(year+1)+'-04-30'
        year_sell = str(year+2)+'-04-30'
        return_year = (ibov_return.loc[year_sell] / ibov_return.loc[year_buy]-1)
        #return_year = return_year.to_frame().reset_index()
        #return_year = return_year.rename(columns = {'index': 'Stock', 0: 'Return'})
    return return_year


def piotroski_return_switch(year, score_long, score_short, long_short, print_result):
    '''
    params:
    year: chose between 2005 and 2017
    cutoff: chose the minimum score for F Score (0-9)
    print_result = 1 then dataframe is printed with results
    print_result = 2 then dataframe is printed without results
    print_result = 3 then only total return is printed
    '''
    
    df = piotroski_rank(year, score_long, score_short, long_short)
    df_pb = price_to_book(year, 0)
    
    find_pb_percentile = find_percentile2(year, 0.2)
    
    if year == 2017:
        year_buy = str(year+1)+'-04-30'
        year_sell = str(year+2)+'-01-31'
        return_year = (close.loc[year_sell] / close.loc[year_buy]-1)
        return_year = return_year.to_frame().reset_index()
        return_year = return_year.rename(columns = {'index': 'Stock', 0: 'Return'})
    
    else:
        year_buy = str(year+1)+'-04-30'
        year_sell = str(year+2)+'-04-30'
        return_year = (close.loc[year_sell] / close.loc[year_buy]-1)
        return_year = return_year.to_frame().reset_index()
        return_year = return_year.rename(columns = {'index': 'Stock', 0: 'Return'})
    
    df2 = pd.merge(pd.merge(df, return_year, on='Stock'), df_pb, on='Stock')
    df2 = df2.dropna()
    
    df3 = df2.copy()
    df3 = df2[df2['Price to Book'] < find_pb_percentile]
    df4 = df3.copy()
    
    df4.loc[df4['F Score'] < score_short, 'Return'] = df4['Return']*-1
    
    total_return = round(df4['Return'].sum()/len(df4['Return']),4)
    number_stocks = len(df4['Return'])
    df4['$ Invested'] = (10000/number_stocks)
    df4['$ win/loss'] = (10000/number_stocks) * df4['Return']
        
    if print_result==1:
    
        print "Portfolio Return for year",year+1,"was:", total_return
        print "The number of stocks on the Portfolio was:",number_stocks
        print "The Price to Book quintile for year",year+1,"was",round(find_pb_percentile,2)
    
        return df4.dropna()
    
    elif print_result==2:
        return df4.dropna()
    
    elif print_result==3:
        return total_return

    elif print_result==4:
        return (total_return, number_stocks)


def piotroski_return_momentum_switch(year, score_long, score_short, threshold_mom, long_short, print_result):
    '''
    params:
    year: chose between 2005 and 2017
    score_long: chose the minimum score for long F Score (0-9)
    score_short: chose the minimum score for short F Score (0-9)
    threshold_mom: chose the minimum return for momentum
    print_result = 1 then dataframe is printed with results
    print_result = 2 then dataframe is printed without results
    print_result = 3 then only total return is printed
    '''
    df = piotroski_return_switch(year, score_long, score_short, long_short, 2)
    
    year_buy = str(year+1)+'-04-30'
    ret6.loc[year_buy]
    return_year = ret6.loc[year_buy]
    
    return_year = return_year.to_frame().reset_index()
    return_year.columns = ['Stock', 'MOM6'] 

    df2 = pd.merge(df, return_year, on='Stock')
    df2 = df2.dropna()
    df3 = df2[df2['MOM6'] > threshold_mom]
    df4 = df3.copy()
            
    total_return = round(df4['Return'].sum()/len(df4['Return']),4)#.format("{:.2%}")
    number_stocks = len(df4['Return'])
    df4['$ Invested'] = (10000/number_stocks)
    df4['$ win/loss'] = (10000/number_stocks) * df4['Return']

    if print_result==1:
        print "Portfolio Return for year",year+1,"was:", total_return
        print "The number of stocks on the Portfolio was:",number_stocks

        return df4.dropna()
    
    elif print_result==2:
        return df4.dropna()
    
    elif print_result==3:
        return total_return

    elif print_result==4:
        return (total_return, number_stocks)


def portfolio_statistcs(year, tipo):
    if year == 2018:
        start = str(year)+'-05'
        end = str(year+1)+'-01'
    else:
        start = str(year)+'-05'
        end = str(year+1)+'-04'
    
    if tipo == 'piotroski':
        stocks_selected = piotroski_return_switch(year-1, 6, 3, 'long', 2)['Stock']
    elif tipo == 'momentum':
        stocks_selected = piotroski_return_momentum_switch(year-1, 6, 3, 0, 'long', 2)['Stock']
    year = stock_return[start:end][stocks_selected]
    year['Portfolio'] = year.sum(axis=1)/len(stocks_selected)
    year['Portfolio_Sum'] = year['Portfolio']+1
    year['Cumulative_Return'] = year['Portfolio_Sum'].cumprod()
    total = year['Cumulative_Return'][-1]-1
    #year['MTSA4_Ret'] = year['VGOR4']+1
    #year['MTSA4_Cum'] = year['MTSA4_Ret'].cumprod()
    
    return year, total


# Calculate all 9 F Score indicators:
# Return on Assets
rank_1_roa = array_to_df(condition_cal(clean_data(df_roa), 'greater'))
# Cash FLow from Operations
rank_2_cfo = array_to_df(condition_cal(clean_data(df_cfo), 'greater'))
# Delta ROA
rank_3_roa_var = array_to_df(condition_cal(clean_data(df_roa), 'diff'))
# Accrual
rank_4_accrual = array_to_df(condition_cal(clean_data(df_accrual), 'greater'))
# Delta Leverage
rank_5_debt = array_to_df(condition_cal(clean_data(df_debt), 'diff', True))
# Delta Liquidity
rank_6_liquidity = array_to_df(condition_cal(clean_data(df_liquidity), 'diff'))
# Delta Equity Offer
rank_7_issue = array_to_df(condition_cal(clean_data(df_stock_issue), 'eq_offer'))
# Delta Margin
rank_8_margin = array_to_df(condition_cal(clean_data(df_margin), 'diff'))
# Delta Turnover
rank_9_turnover = array_to_df(condition_cal(clean_data(df_turnover), 'diff'))

# F Score final calculation:
conso = rank_1_roa + rank_2_cfo + rank_3_roa_var + rank_4_accrual + rank_5_debt + rank_6_liquidity + rank_7_issue + rank_8_margin + rank_9_turnover

total_piotroski_return = []
for i in range(2004, 2018):
    piotroski = piotroski_return_switch(i, 6, 0, 'long', 3)
    total_piotroski_return.append(piotroski)

total_piotroski_mom_return = []
for i in range(2004, 2018):
    piotroski_mom = piotroski_return_momentum_switch(i, 6, 0, 0, 'long', 3)
    total_piotroski_mom_return.append(piotroski_mom)

total_ibov_return = []
for i in range(2004, 2018):
    ibov = ibov_return(close, i)
    total_ibov_return.append(ibov)

piotroski_ret = sum(total_piotroski_return)*100
piotroski_mom_ret = sum(total_piotroski_mom_return)*100
ibov_ret = round(sum(total_ibov_return)*100,2)

if __name__ == "__main__":

    #print conso.tail()
    print "Piotroski Portfolio:" 
    print "Year","      ","No Stocks","  ","Return" 
    for i in range(2004,2018):
        x = piotroski_return_switch(i, 6, 0, 'long', 4)
        y = x[0]
        z = x[1]
        print i+1,".....:",z,".........",y

    print "\n"
    print "="*85

    print "Piotroski with Momentum Portfolio:"
    print "Year","      ","No Stocks","  ","Return"
    for i in range(2004,2018):
        x = piotroski_return_momentum_switch(i, 6, 0, 0, 'long', 4)
        y = x[0]
        z = x[1]
        print i+1,".....:",z,".........",y

    

    print "\n"
    print "="*85
    print "\n Piotroski Portfolio and Return by year - from 2005 to 2018:\n" 
    print "="*85
    #print "\n"

    for i in range(2004,2018):
        x = piotroski_return_switch(i, 6, 0, 'long', 2)
        y = piotroski_return_switch(i, 6, 0, 'long', 3)
        print "Piotroski Portfolio for year:", i, "-->", y
        print x, "\n", "="*85, "\n"
        
    print "\n"
    print "="*85
    print "\n Piotroski Portfolio with Momentum and Return by year - from 2005 to 2018:\n"
    print "="*85
    

    for i in range(2004,2018):
        x = piotroski_return_momentum_switch(i, 6, 0, 0, 'long', 2)
        y = piotroski_return_momentum_switch(i, 6, 0, 0, 'long', 3)
        print "Piotroski Portfolio with Momentum for year:", i, "-->", y
        print x, "\n", "="*85, "\n"

    print "\n"
    print "="*85
    print "Summary Statistics"
    print "="*85

    df = portfolio_returns()
    kpi_output = Output(df)
    print "="*85
    print "\nThese are the KPI's for the 3 portfolios: \n\n", kpi_output.generate_output()
    
    x = portfolio_returns().cumsum()
    x.plot(figsize=(10,6))
    plt.title("Portfolio comparison between 2005 to 2018", weight='bold');
    plt.show()
    
    print "="*85
    print "\nPiotroski Portfolio:" 
    print "Year","      ","No Stocks","  ","Return","       ","Benchmarking Return" 
    for i in range(2004,2018):
        x = piotroski_return_switch(i, 6, 0, 'long', 4)
        ibov = ibov_return(close, i) 
        y = x[0]
        z = x[1]
        print i+1,".....:",z,".........",y,"...............", round(ibov,4) 

    
    print "="*85
    print "\nPiotroski Portfolio with Momentum Return:" 
    print "Year","      ","No Stocks","  ","Return","       ","Benchmarking Return" 
    for i in range(2004,2018):
        x = piotroski_return_momentum_switch(i, 6, 0, 0, 'long', 4)
        ibov = ibov_return(close, i) 
        y = x[0]
        z = x[1]
        print i+1,".....:",z,".........",y,"...............", round(ibov,4) 
    

    print "\n"
    print "="*85
    print "Comparison of Returns"
    print "="*85
    print "Total Piotroski Portfolio Return", "......................:", piotroski_ret, "%"
    print "Total Piotroski Portfolio with Momentum Return", "........:", piotroski_mom_ret, "%"
    print "Total Bovespa (benchmarking) Return", "...................:", ibov_ret, "%"

    
    