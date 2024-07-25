from trainner import train_lstm4pred_singlefeature, train_lstm4pred_multifeature, train_rl
from utils import prepare_data

def test_single_lstm():
    train_lstm4pred_singlefeature('rsi_14','rsi_test', 10,10)
    
def test_multiple_lstm():
    train_lstm4pred_multifeature('vwma_14',["Close","Volume"], ['vwma_14'],10,10)
    
def test_prepare_data():
    pdata = prepare_data()
    pdata.start()


if __name__ == "__main__":
    # test_single_lstm()
    # test_multiple_lstm()
    # test_prepare_data()
    train_rl()