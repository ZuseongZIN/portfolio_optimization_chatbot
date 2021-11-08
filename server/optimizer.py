import datetime
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from scipy.optimize import minimize
import json
from datetime import date
import math
import itertools as it
import operator
from datetime import datetime
from scipy import stats
from scipy.stats import norm
from dateutil import rrule
from calendar import monthrange
from dateutil.relativedelta import relativedelta
from ast import literal_eval
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import sys


#소숫점 표현
pd.options.display.float_format = '{:.3f}'.format
np.set_printoptions(precision=3, suppress=True)

class c_Models:
    #Input 값으로, 자산 list, 사용자 포트폴리오 비중, 시작일, 마지막일
    def __init__(self, assets, assets_w, start, end):
        self.result = None
        self.graph = None
    
        stocks = pd.read_csv('stockcodename.csv', index_col=0)
        symbol = ''
        self.asset_name = assets[:]
        for k in range(len(assets)):
            for i in enumerate(stocks.Name):
                if i[1] == assets[k]:
                    assets[k] = (stocks.iloc[i[0]].Symbol)
                    break

        data = pd.DataFrame()
        # 전체 자산 data들을 가지고 온 후, 정리함
        
        for asset in assets: #total_list:
            tmp = fdr.DataReader(asset,start,end).Close
            if len(data) == 0 :
                data = tmp
            else:
                data = pd.concat([data,tmp], axis=1)
   
        data.columns = self.asset_name
   
        if data.isnull().values.any() == True: #불러온 data에 오류가 있다면
            return "No Data",''

        else:
            data = data.resample('M').mean() #일별 데이터를 월별 데이터로 만들어줌
            data = data.pct_change() #월별 주가 데이터를 이용해 수익률 데이터로 변환
            data.dropna(inplace=True) #결측치 제외(첫 row)

            self.data = data
            self.assets_w = assets_w
            self.mu = data.mean() * 12
            self.cov = data.cov() * 12

    #GMV 최적화 : 제약 조건은 비중합=1, 공매도 불가능
    def gmv_opt(self):
        n_assets = len(self.data.columns)
        w0 = np.ones(n_assets) / n_assets
        fun = lambda w: np.dot(w.T, np.dot(self.cov, w))
        constraints = ({'type':'eq', 'fun':lambda x: np.sum(x)-1})
        bd = ((0,1),) * n_assets
        #cov = data.cov() * 12
        gmv = minimize(fun, w0, method = 'SLSQP', constraints=constraints, bounds=bd)
        result = dict(zip(self.asset_name, np.round(gmv.x,3)))
        return result
    
    #Max Sharp ratio : risk free rate은 0.8%로 지정했고, 
    def ms_opt(self):
        n_assets = len(self.data.columns)
        w0 = np.ones(n_assets) / n_assets
        fun = lambda w: -(np.dot(w.T, self.mu) - 0.008) / np.sqrt(np.dot(w.T, np.dot(self.cov, w)))
        bd = ((0,1),) * n_assets     
        constraints = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
        maxsharp = minimize(fun, w0, method ='SLSQP', constraints=constraints, bounds=bd)
        result = dict(zip(self.asset_name, np.round(maxsharp.x,3)))
        return result
    
    def rp_opt(self):
        def RC(cov, w):
            pfo_std = np.sqrt(np.dot(w.T, np.dot(self.cov, w)))
            mrc = 1/pfo_std * (np.dot(self.cov, w))
            rc = mrc * w
            rc = rc / rc.sum()
            return rc
        
        
        def RP_objective(x):
            pfo_std = np.sqrt(np.dot(x.T, np.dot(self.cov, x)))
            mrc = 1/pfo_std * (np.dot(self.cov, x))
            rc = mrc * x
            rc = rc / rc.sum()

            a = np.reshape(rc, (len(rc),1))
            differs = a - a.T
            objective = np.sum(np.square(differs))

            return objective    
        
        n_assets = len(self.data.columns)
        w0 = np.ones(n_assets) / n_assets
        constraints = [{'type':'eq', 'fun': lambda x: np.sum(x) -1}]
        bd = ((0,1),) * n_assets

        rp = minimize(RP_objective, w0,  constraints=constraints, bounds = bd, method='SLSQP')
        result = dict(zip(self.asset_name, np.round(rp.x,3)))
        return result     #, RC(self.cov, rp.x)

    def plotting(self):
        wt_gmv = np.asarray(list(self.gmv_opt().values()))
        wt_ms = np.asarray(list(self.ms_opt().values()))
        wt_rp = np.asarray(list(self.rp_opt().values()))
        
        ret_gmv = np.dot(wt_gmv, self.mu)
        ret_ms = np.dot(wt_ms, self.mu)
        ret_rp = np.dot(wt_rp, self.mu)
        vol_gmv = np.sqrt(np.dot(wt_gmv.T, np.dot(self.cov, wt_gmv)))
        vol_ms = np.sqrt(np.dot(wt_ms.T, np.dot(self.cov, wt_ms)))
        vol_rp = np.sqrt(np.dot(wt_rp.T, np.dot(self.cov, wt_rp)))
        
        wt_gmv = wt_gmv.tolist()
        wt_ms = wt_ms.tolist()
        wt_rp = wt_rp.tolist()
        
        user_ret = np.dot(self.assets_w, self.mu)
        user_risk = np.sqrt(np.dot(self.assets_w, np.dot(self.cov, self.assets_w)))

        weights = {'gmv': wt_gmv, "ms" : wt_ms, "rp": wt_rp}
        
        #rec_rs = recommended_asset()

        trets = np.linspace(ret_gmv, max(self.mu), 30) # 30개 짜르기 
        tvols = []
        
        efpoints = dict()
        for i, tret in enumerate(trets): #이 개별 return마다 최소 risk 찾기
            n_assets = len(self.data.columns)
            w0 = np.ones(n_assets) / n_assets
            fun = lambda w: np.dot(w.T ,np.dot(self.cov, w))
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                           {'type': 'ineq', 'fun': lambda x: np.dot(x, self.mu) - tret}]
                           #{'type': 'ineq', 'fun': lambda x: x}]
            bd = ((0,1),) * n_assets

            minvol = minimize(fun, w0, method='SLSQP',bounds = bd, constraints=constraints)
            tvols.append(np.sqrt(np.dot(minvol.x, np.dot(self.cov, minvol.x))))
            
            pnumber = '{}point'.format(i+1)
            efpoints[pnumber] = minvol.x.tolist()
        
        if self.data.shape[0] <= 1:
            error = '기간에러'
            return error,1,1
        else:
            ret_vol = {"GMV": [vol_gmv, ret_gmv],"MaxSharp": [vol_ms, ret_ms],"RiskParity": [vol_rp, ret_rp], "Trets" : trets.tolist(), "Tvols": tvols, "User" : [user_risk,user_ret]} #, "Recommended" : rec_rs}        
            return ret_vol, json.dumps(efpoints), json.dumps(weights)


class back_test:
    # 단순 일별수익률의 평균을 *365하여 연간수익률을 산출
    def __init__(self):
        self.test = 0
        
    def Arithmetic_Mean_Annual(self,ret):
        month_return =  np.mean(ret)
        return (month_return*252)

    # 기간중 투자했을때 하락할 수 있는 비율
    def dd(self,ret):
        cum_ret = (1 + ret).cumprod()
        max_drawdown = 0
        max_ret = 1
        dd_list = []
        c = 0
        for ix_ret in cum_ret.values:
            if max_ret < ix_ret:
                max_ret = ix_ret
            dd_list.append((ix_ret - max_ret) / max_ret) 
            c= c+1
        return dd_list
    
    # 기간중 투자했을때 최고로 많이 하락할 수 있는 비율
    def mdd(self,ret):
        
        cum_ret = (1 + ret).cumprod()
        max_drawdown = 0
        max_ret = 1
        for ix_ret in cum_ret.values:
            if max_drawdown > (ix_ret - max_ret) / max_ret:
                max_drawdown = (ix_ret - max_ret) / max_ret
            if max_ret < ix_ret:
                max_ret = ix_ret

        return abs(max_drawdown)

    # 포트폴리오 수익률에서 무위험 수익률을 제한 후 이를 포트폴리오의 표준편차로 나눠 산출한 값, 즉 위험대비 얼마나 수익이 좋은지의 척도
    def sharpe_ratio(self,ret, rf=0.008, num_of_date=252):
        
        return ((np.mean(ret - (rf / num_of_date))) / (np.std(ret))) * np.sqrt(num_of_date)
    
    # 설정한 confidence level에 따른(95%) 확률로 발생할 수 있는 손실액의 최대 액수
    def value_at_risk(self,ret, para_or_hist="para", confidence_level=0.95):
        
        vol = np.std(ret)
        if para_or_hist == "para":
            VaR = np.mean(ret) - vol * norm.ppf(confidence_level)
        else:
            print('error')

        return VaR
    
    # 전체 투자기간에서 상승한 ( ret > 0 ) 기간의 비율
    def winning_rate(self,ret):
        var_winning_rate = np.sum(ret > 0) / len(ret)
        return var_winning_rate    
    
    # 상승한날의 평균상승값을 하락한날의 평균하락값으로 나눈 비율
    def profit_loss_ratio(self,ret):

        if np.sum(ret > 0) == 0:
            var_profit_loss_ratio = 0
        elif np.sum(ret < 0) == 0:
            var_profit_loss_ratio = np.inf
        else:
            win_mean = np.mean(ret[ret > 0])
            loss_mean = np.mean(ret[ret < 0])
            var_profit_loss_ratio = win_mean / loss_mean
        return abs(var_profit_loss_ratio)

    # 데이터 취합하는 코드 
    #임시로 5가지 데이터 예시를 활용해 코드작성
    # 선택한 종목의 이름과 비중, 투자기간을 input 값으로 받음       
    
    def backtest_data(self, assets,weight,start_data_1, end_data_1,start_amount,rebalancing_month, interval, opt_option):
        # input으로 받는 assetnames 입력
        a = assets
        stock_num = len(a)
        # input으로 받는 assetweights 입력
        rebal_month = int(rebalancing_month)
        # input으로 받는 rebalancing_month를 입력
        # 나타내는 데이터 간격을 표시

        # weight 간격  
        b = list(map(float, weight))
        

        # input으로 받는 from_period와 to_period 입력
        stock_return = pd.date_range(start=start_data_1, end=end_data_1)
        stock_return = pd.DataFrame(stock_return)
        stock_return.columns = ['Date']

        stocks = pd.read_csv('stockcodename.csv', index_col=0)
        symbol = ''
        asset_name = assets[:]
        for k in range(len(assets)):
            for i in enumerate(stocks.Name):
                if i[1] == assets[k]:
                    assets[k] = (stocks.iloc[i[0]].Symbol)
                    break
        
        # input으로 받는 from_period와 to_period 입력
        stock_return = pd.date_range(start=start_data_1, end=end_data_1)
        stock_return = pd.DataFrame(stock_return)
        stock_return.columns = ['Date']
        

        for asset in assets: #total_list:
            tmp = fdr.DataReader(asset,start_data_1,end_data_1)
            tmp.insert(1,"Date",tmp.index.copy(),True)
            tmp = tmp[['Date','Change']]
            tmp.columns = ['Date',asset]
            tmp = tmp.reset_index(drop=True)
            stock_return = pd.merge(stock_return,tmp,how='inner', on='Date')

        stock_return = stock_return.dropna(axis=0)

        #print(stock_return)
        if opt_option == 'basic' :

            # 투자비중으로 이루어진 dataframe 만들기

            start_datetime = stock_return.iloc[0,0]
            end_datetime = stock_return.iloc[-1,0]
            diff_months_list = list(rrule.rrule(rrule.MONTHLY, dtstart=start_datetime, until=end_datetime))
            month_gap = len(diff_months_list)
            rebal_roof = month_gap//rebal_month
            rebal_weight = pd.DataFrame()

            for i in range(rebal_roof+1):
                # 데이터로부터 리밸런싱기간만큼 가져오기
                filtered_df =stock_return.loc[stock_return["Date"].between(start_datetime, 
                                                                         start_datetime + relativedelta(months=rebal_month)+relativedelta(days = -1))]
                # 리밸런싱 기간의 누적수익률 산출
                for j in range(stock_num):
                    filtered_df.iloc[:,j+1] = (1 + filtered_df.iloc[:,j+1]).cumprod()
                # 해당 누적수익률에 initial 투자비중을 곱해준다 
                for j in range(stock_num):
                    filtered_df.iloc[:,j+1] = filtered_df.iloc[:,j+1]*float(b[j])
                # 이후 각각의 종목의 비중을 계산해서 산출한다
                filtered_df['total_value'] = filtered_df.sum(axis=1)
                for j in range(stock_num):
                    filtered_df.iloc[:,j+1] = filtered_df.iloc[:,j+1]/filtered_df['total_value']

                rebal_weight = pd.concat([rebal_weight,filtered_df])
                start_datetime = start_datetime + relativedelta(months=rebal_month)

                #final_day = monthrange(start_datetime.year, start_datetime.month)

            stock_weight = rebal_weight.iloc[:,:-1]
            #print(stock_weight)
            '''
            stock_weight = stock_return.Date
            stock_weight = pd.DataFrame(stock_weight)
            c = 0
            for stockweight in b:
                stock_weight[a[c]] = float(stockweight)
                c = c + 1
            #print(stock_weight)
            '''
        else :
            # 포트폴리오 최적화 코드를 통한 리벨런싱 이중 리스트 weight 산출
            # 1. 입력 받은 start ~ end 날짜를 리밸런싱 기간으로 쪼개기   
            opt_start_datetime = stock_return.iloc[0,0]
            opt_end_datetime = stock_return.iloc[-1,0]
            opt_diff_months_list = list(rrule.rrule(rrule.MONTHLY, dtstart=opt_start_datetime, until=opt_end_datetime))
            opt_month_gap = len(opt_diff_months_list)
            opt_rebal_roof = opt_month_gap//rebal_month
            opt_rebal_weight = pd.DataFrame()
            #opt_array = [[0]*stock_num]*(opt_rebal_roof+1)

            for i in range(opt_rebal_roof+1):
                opt_df = stock_return.loc[stock_return["Date"].between(opt_start_datetime,opt_start_datetime + relativedelta(months=rebal_month)+relativedelta(days = -1))]
                # 최적화 코드에서 기간마다의 가중치를 가져온다
                c_m = c_Models(a,b,opt_df.iat[0,0]- relativedelta(months=3),opt_df.iat[-1,0])
                ret_vol, efpoints, weights = c_m.plotting()
                weights = literal_eval(weights)
                weights = weights.get(opt_option)
                ##print(weights)
                # 리밸런싱 기간의 누적수익률 산출
                for j in range(stock_num):
                    opt_df.iloc[:,j+1] = (1 + opt_df.iloc[:,j+1]).cumprod()
                # 해당 누적수익률에 initial 투자비중을 곱해준다 
                for j in range(stock_num):
                    opt_df.iloc[:,j+1] = opt_df.iloc[:,j+1]*float(weights[j])
                # 이후 각각의 종목의 비중을 계산해서 산출한다
                opt_df['total_value'] = opt_df.sum(axis=1)
                for j in range(stock_num):
                    opt_df.iloc[:,j+1] = opt_df.iloc[:,j+1]/opt_df['total_value']

                # 이후 각각의 종목의 비중을 계산해서 산출한다
                #print(opt_df)
                opt_rebal_weight = pd.concat([opt_rebal_weight,opt_df])
                opt_start_datetime = opt_start_datetime + relativedelta(months=rebal_month)
                #리밸런싱으로 start 기간이 고객이 원하는 end 기간보다 커지게 되면 종료 
                if opt_start_datetime > stock_return.iloc[-1,0]:    # i가 100일 때
                    break    
            stock_weight = opt_rebal_weight.iloc[:,:-1]
            ##print(stock_weight)
        # 수익률 데이터와 투자비중을 곱한 하나의 데이터 생성 
        pfo_return = stock_weight.Date
        pfo_return = pd.DataFrame(pfo_return)
        # weight 와 return의 날짜 맞춰주기 
        #pfo_return = pfo_return[0:len(stock_weight)]
        pfo_return = pd.merge(pfo_return, stock_return, left_on='Date', right_on='Date', how='left')
        pfo_return['mean_return'] = 0
        ##print(pfo_return)
        for i in range(0,len(pfo_return)):
            return_result = list(pfo_return.iloc[i,1:1+stock_num])
            return_weight = list(stock_weight.iloc[i,1:1+stock_num])
            pfo_return.iloc[i,1+stock_num]  = np.dot(return_result,return_weight)
            #rint(pfo_return)
        pfo_return['acc_return'] = [x+1 for x in pfo_return['mean_return']]
        pfo_return['acc_return'] = list(it.accumulate(pfo_return['acc_return'], operator.mul))
        pfo_return['acc_return'] = [x-1 for x in pfo_return['acc_return']]
        pfo_return['final_balance'] = float(start_amount) + float(start_amount)*pfo_return['acc_return']
        pfo_return['Drawdown_list'] = back_test.dd(input,pfo_return['mean_return'])
        pfo_return = pfo_return.set_index('Date') 
        #print(pfo_return)
        
        
        ### 벤치마크 데이터 로드 및 전처리
        
        tiker_list = ['KS11','US500'] 
        bench_list = [fdr.DataReader(ticker, start_data_1,  end_data_1)['Change'] for ticker in tiker_list]
        bench = pd.concat(bench_list, axis=1)
        bench.columns = ['KOSPI', 'S&P500']
        bench['KOSPI'] = bench['KOSPI'].fillna(0)
        bench['S&P500'] = bench['S&P500'].fillna(0)
        #bench = bench.dropna()
        
        # 벤치마크 누적수익률, DD 값 
        
        bench['KOSPI_acc'] = [x+1 for x in bench['KOSPI']]
        bench['KOSPI_acc'] = list(it.accumulate(bench['KOSPI_acc'], operator.mul))
        bench['KOSPI_acc'] = [x-1 for x in bench['KOSPI_acc']]
        bench['KOSPI_balance'] = float(start_amount) + float(start_amount)*bench['KOSPI_acc']
        bench['KOSPI_Drawdown'] = back_test.dd(input,bench['KOSPI'])
        bench['S&P500_acc'] = [x+1 for x in bench['S&P500']]
        bench['S&P500_acc'] = list(it.accumulate(bench['S&P500_acc'], operator.mul))
        bench['S&P500_acc'] = [x-1 for x in bench['S&P500_acc']]
        bench['S&P500_balance'] = float(start_amount) + float(start_amount)*bench['S&P500_acc']
        bench['S&P500_Drawdown'] = back_test.dd(input,bench['S&P500'])
        
        if interval == 'monthly' or interval == 'weekly' :
            if interval == 'monthly' :
                inter = 'M'
            if interval == 'weekly' :
                inter = 'W'
            pfo_return_interval = pfo_return.resample(inter).last()
            pfo_return_first = pd.DataFrame(pfo_return.iloc[0]).transpose()
            pfo_return_interval = pd.concat([pfo_return_first, pfo_return_interval])
            pfo_return_interval['mean_return'] = pfo_return_interval['final_balance'].pct_change()
            pfo_return_interval = pfo_return_interval.dropna()
            
            # 월별 간격으로 만들어주기, 여기서는 return과 value만 monthly로 산출함 나머지값은 daily
            bench_interval = bench.resample(inter).last()
            #bench_ex['KOSPI'] = bench_ex['final_balance'].pct_change()
            bench_first = pd.DataFrame(bench.iloc[0]).transpose()
            bench_interval = pd.concat([bench_first, bench_interval])
            bench_interval['KOSPI'] = bench_interval['KOSPI_balance'].pct_change()
            bench_interval['S&P500'] = bench_interval['S&P500_balance'].pct_change()
            bench_interval = bench_interval.dropna()
            
            # 날짜타입 열로 만들기 및 str 타입으로 전처리 
            pfo_return = pfo_return.rename_axis('Date').reset_index()
            pfo_return['Date'] =  pd.to_datetime(pfo_return['Date'], format='%d/%m/%Y').dt.date
            pfo_return['Date'] = list(map(str, pfo_return['Date']))
            
            pfo_return_interval = pfo_return_interval.rename_axis('Date').reset_index()
            pfo_return_interval['Date'] =  pd.to_datetime(pfo_return_interval['Date'], format='%d/%m/%Y').dt.date
            pfo_return_interval['Date'] = list(map(str, pfo_return_interval['Date']))
            
            bench = bench.rename_axis('Date').reset_index()
            bench['Date'] =  pd.to_datetime(bench['Date'], format='%d/%m/%Y').dt.date
            bench['Date'] = list(map(str, bench['Date']))        
           
            bench_interval = bench_interval.rename_axis('Date').reset_index()
            bench_interval['Date'] =  pd.to_datetime(bench_interval['Date'], format='%d/%m/%Y').dt.date
            bench_interval['Date'] = list(map(str, bench_interval['Date']))       
            
            backtest_return = {
                 'pfo_return': [
                         {
                         'Date': list(pfo_return_interval['Date']),
                         'mean_return': list(pfo_return_interval['mean_return']),                 
                         'acc_return ratio': list(pfo_return_interval['acc_return']),
                         'final_balance': list(pfo_return_interval['final_balance']),
                         'Drawdown_list' : list(pfo_return_interval['Drawdown_list'])
                          }
                 ],         
                 'bench': [
                         {
                         'Date': list(bench_interval['Date']),
                         'KOSPI_return': list(bench_interval['KOSPI']),              
                         'S&P500_return': list(bench_interval['S&P500']),
                         'KOSPI_acc_return': list(bench_interval['KOSPI_acc']),
                         'KOSPI_balance' : list(bench_interval['KOSPI_balance']),                 
                         'KOSPI_Drawdown': list(bench_interval['KOSPI_Drawdown']),
                         'S&P500_acc_return': list(bench_interval['S&P500_acc']),
                         'S&P500_balance' : list(bench_interval['S&P500_balance']),                 
                         'S&P500_Drawdown': list(bench_interval['S&P500_Drawdown'])
                          }
                 ],    
                 'indicator': [
                         {
                         'Mean': back_test.Arithmetic_Mean_Annual(input,pfo_return['mean_return']),
                         'Std': pfo_return['mean_return'].std() * np.sqrt(365),                 
                         'Sharpe ratio': back_test.sharpe_ratio(input,pfo_return['mean_return']),
                         'VaR': back_test.value_at_risk(input,pfo_return['mean_return']),
                         'MDD': back_test.mdd(input,pfo_return['mean_return']),
                         'Winning ratio': back_test.winning_rate(input,pfo_return['mean_return']),
                         'Gain/Loss Ratio': back_test.profit_loss_ratio(input,pfo_return['mean_return'])
                          }
                 ],    
                 'KOSPI_indicator': [
                         {
                         'Mean': back_test.Arithmetic_Mean_Annual(input,bench['KOSPI']),
                         'Std': bench['KOSPI'].std() * np.sqrt(365),                 
                         'Sharpe ratio': back_test.sharpe_ratio(input,bench['KOSPI']),
                         'VaR': back_test.value_at_risk(input,bench['KOSPI']),
                         'MDD': back_test.mdd(input,bench['KOSPI']),
                         'Winning ratio': back_test.winning_rate(input,bench['KOSPI']),
                         'Gain/Loss Ratio': back_test.profit_loss_ratio(input,bench['KOSPI'])
                          }
                 ],    
                 'S&P500_indicator': [
                         {
                         'Mean': back_test.Arithmetic_Mean_Annual(input,bench['S&P500']),
                         'Std': bench['S&P500'].std() * np.sqrt(365),                 
                         'Sharpe ratio': back_test.sharpe_ratio(input,bench['S&P500']),
                        'VaR': back_test.value_at_risk(input,bench['S&P500']),
                         'MDD': back_test.mdd(input,bench['S&P500']),
                         'Winning ratio': back_test.winning_rate(input,bench['S&P500']),
                         'Gain/Loss Ratio': back_test.profit_loss_ratio(input,bench['S&P500'])
                          }
                 ]
             } 
            
        else :
            # 날짜타입 열로 만들기 및 str 타입으로 전처리 
            pfo_return = pfo_return.rename_axis('Date').reset_index()
            pfo_return['Date'] =  pd.to_datetime(pfo_return['Date'], format='%d/%m/%Y').dt.date
            pfo_return['Date'] = list(map(str, pfo_return['Date']))
            
            bench = bench.rename_axis('Date').reset_index()
            bench['Date'] =  pd.to_datetime(bench['Date'], format='%d/%m/%Y').dt.date
            bench['Date'] = list(map(str, bench['Date']))
            backtest_return = {
                 'pfo_return': [
                         {
                         'Date': list(pfo_return['Date']),
                         'mean_return': list(pfo_return['mean_return']),                 
                         'acc_return ratio': list(pfo_return['acc_return']),
                         'final_balance': list(pfo_return['final_balance']),
                         'Drawdown_list' : list(pfo_return['Drawdown_list'])
                          }
                 ],         
                 'bench': [
                         {
                         'Date': list(bench['Date']),
                         'KOSPI_return': list(bench['KOSPI']),              
                         'S&P500_return': list(bench['S&P500']),
                         'KOSPI_acc_return': list(bench['KOSPI_acc']),
                         'KOSPI_balance' : list(bench['KOSPI_balance']),                 
                         'KOSPI_Drawdown': list(bench['KOSPI_Drawdown']),
                         'S&P500_acc_return': list(bench['S&P500_acc']),
                         'S&P500_balance' : list(bench['S&P500_balance']),                 
                         'S&P500_Drawdown': list(bench['S&P500_Drawdown'])
                          }
                 ],    
                 'indicator': [
                         {
                         'Mean': back_test.Arithmetic_Mean_Annual(input,pfo_return['mean_return']),
                         'Std': pfo_return['mean_return'].std() * np.sqrt(365),                 
                         'Sharpe ratio': back_test.sharpe_ratio(input,pfo_return['mean_return']),
                         'VaR': back_test.value_at_risk(input,pfo_return['mean_return']),
                         'MDD': back_test.mdd(input,pfo_return['mean_return']),
                         'Winning ratio': back_test.winning_rate(input,pfo_return['mean_return']),
                         'Gain/Loss Ratio': back_test.profit_loss_ratio(input,pfo_return['mean_return'])
                          }
                 ],    
                 'KOSPI_indicator': [
                         {
                         'Mean': back_test.Arithmetic_Mean_Annual(input,bench['KOSPI']),
                         'Std': bench['KOSPI'].std() * np.sqrt(365),                 
                         'Sharpe ratio': back_test.sharpe_ratio(input,bench['KOSPI']),
                         'VaR': back_test.value_at_risk(input,bench['KOSPI']),
                         'MDD': back_test.mdd(input,bench['KOSPI']),
                         'Winning ratio': back_test.winning_rate(input,bench['KOSPI']),
                         'Gain/Loss Ratio': back_test.profit_loss_ratio(input,bench['KOSPI'])
                          }
                 ],    
                 'S&P500_indicator': [
                         {
                         'Mean': back_test.Arithmetic_Mean_Annual(input,bench['S&P500']),
                         'Std': bench['S&P500'].std() * np.sqrt(365),                 
                         'Sharpe ratio': back_test.sharpe_ratio(input,bench['S&P500']),
                        'VaR': back_test.value_at_risk(input,bench['S&P500']),
                         'MDD': back_test.mdd(input,bench['S&P500']),
                         'Winning ratio': back_test.winning_rate(input,bench['S&P500']),
                         'Gain/Loss Ratio': back_test.profit_loss_ratio(input,bench['S&P500'])
                          }
                 ]
             }  

        return backtest_return

    

# print(back_test().backtest_data(['삼성전자','LG전자'],[0.9,0.1],'2010-01-01', '2021-01-01',10000000,3, 'monthly', 'gmv')['pfo_return'].mean_return)
# print(back_test().backtest_data(['삼성전자','LG전자'],[0.9,0.1],'2010-01-01', '2021-01-01',10000000,3, 'monthly', 'gmv')['pfo_return'][0]['acc_return_ratio'])
# print(back_test().backtest_data(['삼성전자','LG전자'],[0.9,0.1],'2018-01-01', '2021-01-01',10000000,6, 'monthly', 'gmv'))


if (sys.argv[2] == "weight"): 
    args = sys.argv[1].split(",")
    assets = []
    gravity = []
    for i in range(len(args)-1):
        assets.append(args[i])
        gravity.append(0)

    if (sys.argv[3] == 'gmv'):     
        print(c_Models(assets, gravity, args[-1],'2021-06-01').gmv_opt())
    elif (sys.argv[3] == 'ms'):
        print(c_Models(assets, gravity, args[-1],'2021-06-01').ms_opt())
    elif (sys.argv[3] == 'rp'):
        print(c_Models(assets, gravity, args[-1],'2021-06-01').rp_opt())

else:
    args = sys.argv[1].split(",")
    assets = []
    gravity = []
    for i in range(len(args)-1):
        assets.append(args[i])
        gravity.append(0)
    data = back_test().backtest_data(assets,gravity,args[-1], '2021-06-01',10000000,6, 'monthly', sys.argv[3])




#data = back_test().backtest_data(['삼성전자','LG전자','카카오','호텔신라'],[0,0,0,0],'2020-01-01', '2021-01-02',10000000,6, 'monthly', 'ms')
# data = back_test().backtest_data(['삼성전자','LG전자'],[0.1,0.1],'2020-01-01', '2021-01-02',10000000,6, 'monthly', 'gmv')
x = data['pfo_return'][0]['Date']
y = data['pfo_return'][0]['acc_return ratio']
y2 = data['bench'][0]['KOSPI_acc_return']
y3 = data['bench'][0]['S&P500_acc_return']
x_ticks = []
for i,j in enumerate(x):
    if (i % 6) == 0:
        x_ticks.append(j)
    else:
        x_ticks.append('')
x_ticks[-1]= x[-1]
plt.figure(figsize=(10,5))
ax=plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(12))
plt.plot(x,y,label = sys.argv[3] +' result')
plt.plot(x,y2 ,label = 'kospi result')
plt.plot(x,y3, label = 's&p500 result')
plt.xticks(x_ticks,rotation=60)
plt.xlabel('Date')
plt.ylabel('Return')
plt.title('result')
plt.legend()
plt.show()
plt.savefig("./src/test.png", dpi = 100)
print("end")


