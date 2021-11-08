import FinanceDataReader as fdr
import pandas as pd
import sys
from fuzzywuzzy import process

# -*- coding: utf-8 -*-
     
def get_matches(query, choices, limit=3):
    result = process.extract(query, choices, limit=limit)
    return result

def basicinform(input):
    stocks = pd.read_csv('stockcodename.csv', names=['Symbol', 'Market', 'Name'
    , 'Sector', 'Industry', 'ListingDate', 'SettleMonth', 'Represetitive', 'HomePage', 'Region'], index_col=0)
    symbol = ''

    for i in enumerate(stocks.Name):
        if i[1] == input:
            symbol = (stocks.iloc[i[0]].Symbol)
            break

    if(symbol == ''):
        fuzzy = get_matches(input, stocks.Name)
        cand = ''
        for i in fuzzy:
            cand += i[0]
            cand += "\n"      
        cand += "중 찾는게 있으신가요? 다시 입력해주세요."
        return cand

    df = fdr.DataReader(symbol)
    ror_df = df.Close.pct_change()
    volume = df.Volume.iloc[-1]
    price = df.Close.iloc[-1]
    ror = ror_df[-1]

    ror = round(ror, 4)
    ror = ror * 100
    value = ''
    value = "1현재가: " + str(price) + "원\n거래량: " + str(volume) + "건\n전일대비: " + str(ror) + "%"
    # value = {
    #             "현재가": price,
    #             "거래랑": volume,
    #             "전일 대비 수익률:": ror
    #         }
    return value


# print(basicinform('호텔신라'))

args = sys.argv
print(basicinform(args[1]))
