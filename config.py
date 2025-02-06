# TICKET_LIST = ["AAPL","TSLA","NFLX","BNTX","AMZN","MSFT","META","NVDA","GOOGL","PYPL","CSCO","ADBE","INTU",
#         "LI","NEE","ON","NIU","ZM","MMM","DUOL","COIN","AVGO","ABNB","QCOM","LOGI","WDC","DBX","CFLT","AMD","ORCL","CRM","MDB","EBAY",
#         "IBM","BABA","SE","U","SPOT","WAL","RACE","ACN","HPQ","TSM","SBUX","NKE","XOM","DIS","BA","PLTR","SMCI","MSTR","INTC",
#         "CLSK","BAC","AMC","UNH","UBER","COST","PFE","CRWD","OXY","WMT","LLY","MCD","DELL","SNOW","SOFI","ASML","V","CELH","HIMS"
#         "JNJ","PG","ENPH","PEP","MA","ABBV","GRAB","KO","PDD","TMDX","ARM","MU","JPM", "PANW","CHWY","F","JD","MRNA","CVX","RBLX","MRK",
#         "RDDT","BIDU","RTX","PINS","WBD","HSY","HPE","TM","GTLB","NOC","BILI","MDLZ","K","WIX","MSI","NVO","PVH","CDNA","BOX","ANF","AMBA",
#         "S","BHP","TCOM","WDAY","INTU","NTES","A","SNPS","M","ADI","TJX","TGT","COTY","MDT","LOW","FN","EL","ZIM","GLOB", "EA", "RGTI"]

# test
TICKET_LIST = ["AAPL","TSLA","NFLX"]

# ADI
SECTOR_GROUP = {
        "software":["MSFT","NFLX","GOOGL","META","ADBE","ZM","DUOL","COIN","ABNB","CRM","MDB","EBAY",
                    "BABA","U","SPOT","ACN","DIS","RDDT","BIDU","PINS","GTLB","WIX","BOX","AMBA","TCOM",
                    "WDAY","GLOB"],
        "security":["CRWD","PANW","S"],
        "ecommerce":["EBAY","BABA","SE","PDD","JD"],
        "community":["META","GOOGL","ZM","RDDT","MSI"],
        "entertain":["NFLX","AMZN","DIS","AMC","BIDU","WBD","BILI"],
        "cloud":["AMZN","GOOGL","DBX","CFLT","ORCL","IBM","CLSK","SNOW"],
        "chip":["ON","AAPL","NVDA","AVGO","QCOM","AMD","TSM","SMCI","INTC","ASML","ARM","SNPS","RGTI"],
        "industries":["FN"],
        "hardware":["LOGI","AMZN","MSFT","WDC","HPQ","INTC","AAPL","NVDA","AMD","DELL","MU","HPE","MSI"],
        "auto":["TSLA","LI","NIU","RACE","F","TM"],
        "bio":["BNTX"],
        "finance":["PYPL","INTU","COIN","WAL","BAC","SOFI","V","MA","JPM","INTU"],
        "network":["CSCO"],
        "energy":["NEE","XOM", "OXY", "ENPH","CVX","BHP"],
        "utility":["MMM","AMZN","NKE","COST","WMT","PG","PVH","ANF","M","TJX","COTY","LOW","EL"],
        "phone":["AAPL", "GOOGL"],
        "study":["DUOL"],
        "delivery":["UBER","GRAB","TGT","ZIM"],
        "crypto":["COIN","MSTR","CLSK"],
        "rent":["ABNB","TCOM"],
        "food":["SBUX","MCD","CELH","PEP","KO","HSY","MDLZ","K"],
        "pet":["CHWY"],
        "aircraft":["BA"],
        "game":["RBLX","U","EA","NTES"],
        "health":["UNH","PFE","LLY","HIMS","JNJ", "ABBV","TMDX","MRNA","MRK","NVO","CDNA","A","MDT"],
        "astros":["RTX","NOC"]
}

INDICATOR_LIST = ['rsi_14','stochrsi_14','vwma_20','ema_200', 'ema_50', 'ema_100', 'macd', 'ichimoku']


MAP_COLUMNS_NAME = {
    "Close":"close",
    "High":"high",
    "Low":"low",
    "Open":"open",
    "Volume":"volume"
}

# config indicator
RSI_UP = 0.7
RSI_DOWN = 0.3
STORSI_UP = 0.8
STORSI_DOWN = 0.2
ICHIMOKU_UP = 1
ICHIMOKU_DOWN = 0.5

# end year for get data
YEAR_END = 5


