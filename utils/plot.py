import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_hits(series_data):
    os.makedirs(os.path.join(os.getcwd(),'plot_image'),exist_ok=True)
    series_data.hist(bins=30)  # หรือใช้จำนวน bins ที่เหมาะสมกับข้อมูล
    plt.title('Histogram')
    plt.savefig(os.path.join(os.getcwd(),'plot_image',"plot_hits.jpg"))

def plot_kde(series_data):
    os.makedirs(os.path.join(os.getcwd(),'plot_image'),exist_ok=True)
    sns.histplot(series_data, kde=True)
    plt.title("image_kde")
    plt.savefig(os.path.join(os.getcwd(),'plot_image',"plot_kde.jpg"))