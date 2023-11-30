import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DataImporter:
    def __init__(self, data_path = "../Final-IRONHACK-Project/data/"):
        self.data_path = data_path

    def import_data(self, file_name):
        file_path = f"{self.data_path}{file_name}.csv"
        df = pd.read_csv(file_path)
        return df
    

class Explorer:
    def __init__(self,df):
        self.df = df

    def exploration(self):
        print(self.df.isna().sum())

        for i in self.df:
            if i == "label":
                print(self.df["label"].value_counts())
    
    def drop_nulls(self):
        self.df.dropna(inplace = True)

    def describe_data(self):
        print(self.df.describe())

    def correlation_analysis(self):
        new_df = self.df[["label", "center_id"]]
        new_df = pd.get_dummies(new_df, columns = ["label"])
        c = new_df.corr()
        return c
            

class Visualization:
    def __init__(self, df):
        self.df = df
    
    def plot_labels(self):
        plt.figure(figsize=(10, 6))
        sns.countplot(x=self.df["label"], hue=self.df["label"], 
                        palette=["#807dba", "#54278f"])
        sns.despine(offset=10, trim=True)
        plt.show()

    def plot_labels_spec(self):
        plt.figure(figsize=(10, 6))
        sns.countplot(x=self.df["other_specified"], hue=self.df["other_specified"],
                        palette=["#807dba", "#54278f"])
        sns.despine(offset=10, trim=True)
        plt.show()

    def plot_center(self):
        plt.figure(figsize=(10, 6))
        sns.countplot(x=self.df["center_id"], hue=self.df["label"],
                        palette=["#807dba", "#54278f"])
        sns.despine(offset=10, trim=True)
        plt.show()

    def heatmaps(self, c):
        mask = np.triu(np.ones_like(c, dtype=bool), k = 1)
        sns.heatmap(c, mask = mask, annot= True, square=True,
                    cmap = sns.cubehelix_palette(start=2.7, as_cmap=True),
                    linewidths=.5)



    