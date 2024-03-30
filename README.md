# 強化学習の勉強用
強化学習を勉強するためのリポジトリ。<br>
参考書籍は以下の通り。

- [強化学習（第2版）](https://amzn.asia/d/aZi6VdS)
- [強化学習 (機械学習プロフェッショナルシリーズ)](https://amzn.asia/d/hc2QY1E)

# バンディット問題
## 定常バンディット問題
各アームの報酬分布が時間によって、変化しないもの。<br>

例えば、こんな分布。
![](imgs/img_bandit/GaussianBanditEnv.png)

epsilon greedy法で学習するとこんな感じ。<br>
epsilonは0, 0.01, 0.1で試した。
![](imgs/img_bandit/reward_plot.png)
このグラフは、2000個のバンディット問題に対して、各ステップごとに報酬を平均した時の時間変化を示したグラフ。<br>
探索の大事さがよくわかる。<br>