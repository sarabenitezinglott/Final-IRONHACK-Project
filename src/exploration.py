import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DataImporter:
    def __init__(self, data_path = "D:/bootcamp/"):
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

class Labeling_images:
    def __init__(self, df):
        self.df = df 

    def class_train(self):
        # Images addresses
        self.df["file_path"] = self.df["image_id"].apply(lambda x: "D:/bootcamp/original/" + x + ".tif")
        return self.df

    def class_test(self):
        self.df["file_path"]  = self.df["image_id"].apply(lambda x: "D:/bootcamp/test/" + x + ".tif")
        return self.df

    def CElabels(self):
        # Labels: binary classification (0 False, 1 True)
        self.df["Y"] = self.df["label"].apply(lambda x : 1 if x=="CE" else 0) 
        return self.df 
    
    def class_again_train(self):
        self.df["new_file_path"] = self.df["image_id"].apply(lambda x: "D:/bootcamp/original/train_folder/" + x + ".tif")
        self.df.drop(columns = ["file_path"], inplace = True)
        return self.df
    
    def class_again_val(self):
        self.df["new_file_path"] = self.df["image_id"].apply(lambda x: "D:/bootcamp/original/val_folder1/" + x + ".tif")
        self.df.drop(columns = ["file_path"], inplace = True)
        return self.df

    