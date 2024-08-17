from trainner import train_lstm4pred_singlefeature, train_lstm4pred_multifeature, train_rl
from utils import prepare_data

pdata = prepare_data()

def test_single_lstm():
    train_lstm4pred_singlefeature('rsi_14','rsi_test', 10,10)
    
def test_multiple_lstm():
    train_lstm4pred_multifeature('vwma_14',["Close","Volume"], ['vwma_14'],10,10)
    
def test_collect_data():
    data = pdata.collect_data()
    return data

def test_fill_missing_predict(data):
    data = pdata.filling_missing_value(data)
    return data

def test_fill_missing(data):
    data = pdata.filling_missing_value(data, predict_missing= "remove_nan")
    return data

if __name__ == "__main__":
    # test_single_lstm()
    test_multiple_lstm()
    # test_multiple_lstm
    # test_prepare_data()
    # train_rl()
    
    # data = test_collect_data()
    # data = test_fill_missing_predict(data)
    # print(data)
    