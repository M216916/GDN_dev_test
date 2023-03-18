# モデル
### GDN_dev_test1 （test用）
* 提案モデル（full, freeze, onlytime, nontime）のみ実行


# 概要

* データを学習用とテスト用に分割しtestを行う

* ハイパーパラメータは GDN_dev_optuna（ https://github.com/M216916/GDN_dev_optuna ）で決定する

# 環境
* M216916/GDN_dev_optuna/README_2.md と同様
 
 
# データ
* yfinance_01_10

        時系列属性：2730社 * 日次データ（2614日）

        非時系列属性：40カラム（財務指標14+属性26／標準化あり）

        10社分のデータ（主に動作を確認するために使用）

* yfinance_01_100

        100社分のデータ（実験用のデータ）

* yfinance_01_all

        すべて（2730社分）のデータ（本当はこれで実験したかったが時間がかかるため断念）


# 調整事項

* main.py

  データの分割を指定

  時系列属性の平均化（圧縮）の調整

  実行モデルの選択
  ⇒model_flag = ['full', 'freeze', 'onlytime', 'nontime']から選択

* run.sh

  loss_functionの選択（CE_loss or Dice_loss）

  エポック数の調整（pre_EPOCH and fin_EPOCH）

  その他ハイパーパラメータの調整（GDN_dev_optuna で）
