# -*- coding: utf-8 -*-
# 作者 ：LuGang
# 开发时间 ：2021/8/8 17:12
# 文件名称 ：data_process.py
# 开发工具 ：PyCharm


import numpy as np
import pandas as pd


class GetOrigData:
    def __init__(self):
        pass

    def GetUserData(self):
        orig_user_data = pd.read_table("users.dat", sep="::", engine="python",
                                       names=["userid", "gender", "age", "occupation", "zipcode"])
        # 删去zip-code列
        del orig_user_data["zipcode"]
        # one-hot编码
        '''orig_user_data = orig_user_data.join(pd.get_dummies(orig_user_data.gender,
                                                            drop_first=True))
        del orig_user_data["gender"]'''

        return orig_user_data

    def GetMovieData(self):
        orig_movie_data = pd.read_table("movies.dat", sep="::", engine="python",
                                        names=["movieid", "title", "genres"])
        # 删掉电影名列
        del orig_movie_data["title"]
        return orig_movie_data

    def GetRatingData(self):
        orig_rating_data = pd.read_table("ratings.dat", sep="::", engine="python",
                                         names=["userid", "movieid", "rating", "timestamp"])
        # 删掉时间戳列
        del orig_rating_data["timestamp"]
        return orig_rating_data

    # 将用户特征和电影特征进行拼接
    def GetData(self):
        orig_user_data = self.GetUserData()
        orig_movie_data = self.GetMovieData()
        orig_rating_data = self.GetRatingData()

        orig_data_path = "orig_data_xy.csv"
        n = orig_rating_data.shape[0]
        for i in range(n):
            userid = orig_rating_data.iloc[i, 0]
            movieid = orig_rating_data.iloc[i, 1]
            rating = [orig_rating_data.iloc[i, 2]]
            user_featrue = orig_user_data[orig_user_data.userid == userid].values[0].tolist()[1:]
            movie_featrue = orig_movie_data[orig_movie_data.movieid == movieid].values[0].tolist()[1:]
            row = user_featrue + movie_featrue + rating
            row = pd.DataFrame(columns=row)
            row.to_csv(orig_data_path, index=False, mode="a", encoding="utf-8")
            if i % 10000 == 0:
                print("%f percent loaded" % (i * 100 / n))

        GetFinalData()
        print("数据创建成功！！！")

    # 划分训练集和测试集
    def split_train_test_data(self):
        split_rate = 0.7  # 70% 的数据作为训练集
        datapath = "orig_data.csv"
        orig_data = pd.read_csv(datapath, sep=",", header=0)
        n = orig_data.shape[0]
        # 随机打乱
        orig_data = orig_data.sample(n=n, random_state=2)
        orig_data = orig_data.reset_index(drop=True)
        train_index = int(split_rate * n)
        orig_data.loc[0:train_index, :].to_csv("train_data.csv")
        orig_data.loc[train_index + 1:n, :].to_csv("test_data.csv")
        print("训练集、测试集划分完成！！")


def GetFinalData():
    datapath = "orig_data_xy.csv"
    # 表头
    names = ["gender", "age", "occupation", "genres", "rating"]
    orig_data = pd.read_csv(datapath, sep=",", header=None, names=names)
    all_genres = ["Action", "Adventure", "Animation", "Children's", "Comedy",
                  "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
                  "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    values = pd.DataFrame(np.zeros((len(orig_data), len(all_genres))))
    orig_data[all_genres] = values

    for i in range(orig_data.shape[0]):
        a = orig_data.loc[i, "genres"].split("|")
        for g in a:
            orig_data.loc[i, g] = 1

        if i % 5000 == 0:
            print(i)

    col = orig_data.loc[:, "rating"].values.tolist()
    del orig_data["genres"]
    del orig_data["rating"]
    orig_data["rating"] = col
    orig_data.to_csv("orig_data.csv")


if __name__ == "__main__":
    data = GetOrigData()
    data.GetData()
    data.split_train_test_data()

