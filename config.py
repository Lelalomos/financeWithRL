# TICKET_LIST = ["AAPL","TSLA","NFLX","BNTX","AMZN","MSFT","META","NVDA","GOOGL","PYPL","CSCO","ADBE","INTU",
#         "LI","NEE","ON","NIU","ZM","MMM","DUOL","COIN","AVGO","ABNB","QCOM","LOGI","WDC","DBX","CFLT","AMD","ORCL","CRM","MDB","EBAY",
#         "IBM","BABA","SE","U","SPOT","WAL","RACE","ACN","HPQ","TSM","SBUX","NKE","XOM","DIS","BA","PLTR","SMCI","MSTR","INTC",
#         "CLSK","BAC","AMC","UNH","UBER","COST","PFE","CRWD","OXY","WMT","LLY","MCD","DELL","SNOW","SOFI","ASML","V","CELH","HIMS"
#         "JNJ","PG","ENPH","PEP","MA","ABBV","GRAB","KO","PDD","TMDX","ARM","MU","JPM", "PANW","CHWY","F","JD","MRNA","CVX","RBLX","MRK",
#         "RDDT","BIDU","RTX","PINS","WBD","HSY","HPE","TM","GTLB","NOC","BILI","MDLZ","K","WIX","MSI","NVO","PVH","CDNA","BOX","ANF","AMBA",
#         "S","BHP","TCOM","WDAY","INTU","NTES","A","SNPS","M","ADI","TJX","TGT","COTY","MDT","LOW","FN","EL","ZIM","GLOB", "EA", "RGTI"]

# test
TICKET_LIST = ["GOOGL","MSFT","META","NFLX","RDDT","ABNB"]

SECTOR_GROUP = {
        "software":["ADBE","DUOL","CRM","MDB",
                    "U","SPOT","ACN","PINS","GTLB","WIX","BOX","AMBA",
                    "WDAY","GLOB","MSFT","GOOGL"],
        "security":["CRWD","PANW","S"],
        "ecommerce":["EBAY","BABA","SE","PDD","JD"],
        "community":["META","ZM","RDDT"],
        "entertain":["NFLX","DIS","AMC","BIDU","WBD","BILI"],
        "cloud":["DBX","CFLT","ORCL","IBM","CLSK","SNOW"],
        "chip":["ON","AVGO","QCOM","AMD","TSM","SMCI","INTC","ASML","ARM","SNPS","RGTI"],
        "industries":["FN"],
        "hardware":["LOGI","WDC","HPQ","NVDA","DELL","MU","HPE","MSI"],
        "auto":["TSLA","LI","NIU","RACE","F","TM"],
        "bio":["BNTX"],
        "finance":["PYPL","INTU","WAL","BAC","SOFI","V","MA","JPM"],
        "network":["CSCO"],
        "energy":["NEE","XOM", "OXY", "ENPH","CVX","BHP"],
        "utility":["MMM","NKE","COST","WMT","PG","PVH","ANF","M","TJX","COTY","LOW","EL"],
        "phone":["AAPL"],
        "study":["DUOL"],
        "delivery":["UBER","GRAB","TGT","ZIM","AMZN"],
        "crypto":["COIN","MSTR"],
        "rent":["ABNB","TCOM"],
        "food":["SBUX","MCD","CELH","PEP","KO","HSY","MDLZ","K"],
        "pet":["CHWY"],
        "aircraft":["BA"],
        "game":["RBLX","U","EA","NTES"],
        "health":["UNH","PFE","LLY","HIMS","JNJ", "ABBV","TMDX","MRNA","MRK","NVO","CDNA","A","MDT"],
        "astros":["RTX","NOC"]
}

MODEL = "lstm_with_attention"

WEIGHT_GROUP = {
    "software":1,
    "security":1,
    "ecommerce":1,
    "community":1,
    "entertain":1,
    "cloud":1,
    "chip":1,
    "industries":1,
    "hardware":1,
    "auto":1,
    "bio":1,
    "finance":1,
    "network":1,
    "energy":1,
    "utility":1,
    "phone":1,
    "study":1,
    "delivery":1,
    "crypto":1,
    "rent":1,
    "food":1,
    "pet":1,
    "aircraft":1,
    "game":1,
    "health":1,
    "astros":1
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

# lstm model config
LSTM_PARAMS = {'output_size': 226, 'embedding_dim_stock': 6, 'embedding_dim_group': 10, 'embedding_dim_day': 44, 'embedding_dim_month': 18, 'first_layer_hidden_size': 182, 'first_layer_size': 1, 'second_layer_hidden_size': 376, 'second_layer_size': 1, 'third_layer_hidden_size': 216, 'third_layer_size': 1, 'dropout': 0.23667559523715945, 'delta': 0.10007115305567968}

