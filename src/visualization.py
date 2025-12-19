import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(df):
    sns.countplot(x="Class", data=df, palette="coolwarm")
    plt.title("Fraud vs Legit Transaction Distribution")
    plt.show()