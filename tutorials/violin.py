
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("titanic")
# sns.catplot(data=df, x="age", y="class")


# sns.catplot(
#     data=df, x="age", y="class", hue="sex",
#     kind="violin", bw=.25, cut=0, split=True,
# )

sns.catplot(
    data=df, x="class", y="age", hue="sex",
    kind="violin", bw=.25, cut=0, split=True,
)

plt.show()


 # ax4 = sns.catplot(x="elo_cats",
#                   y="COL",
#                   hue="won_lost",
#                   data=df,
#                   kind="violin",
#                   split=True,
#                   dodge=True,
#                   legend=True,
#                   orient='h'
#                   )
#