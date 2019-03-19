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
        df = price_to_book(year,0)
        df2 = df['Price to Book'].quantile(quantile)
        return df2


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


def ibov_return(data, year):
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

    
    