# 第一次出预测演练 | First-forecast drill: from install to a verified forecast plot

把「装好 Kronos、出一张能核对的预测图」这条链路一次走通，顺序别乱（先小档、短窗口，验完再放大）：

1. **装环境**：建 Python 3.10+ 环境，克隆原仓，`pip install -r requirements.txt`。先不碰微调，推理链路跑通再说。
2. **选最小档起步**：先用 Kronos-small（24.7M）配 Kronos-Tokenizer-base——tokenizer 与模型必须配套（mini 才用 Tokenizer-2k），配错一边是新手最常见的报错来源。
3. **备一段你熟的数据**：任选一个你有历史 K 线的标的，导出 CSV，列至少要有 `open/high/low/close`（`volume/amount` 可选），外加一列时间戳。选你自己能肉眼判断走势的一段，别拿陌生数据验模型。
4. **按五步出预测**：载 tokenizer 与模型 → 建 `KronosPredictor(model, tokenizer, max_context=512)` → 切 lookback 400 根作 `df` 与 `x_timestamp`、其后 120 根的时间戳作 `y_timestamp`（lookback 不许超过 512）→ `predict(pred_len=120, T=1.0, top_p=0.9, sample_count=1)`。原仓 `examples/prediction_example.py` 有带画图的完整脚本，可以对照。
5. **和真实走势对图核验**：因为你手里有其后 120 根的真实数据，把预测序列与真实 OHLC 画在同一张图上，先看方向命中率，再看幅度偏差——这一步就是你的迷你样本外检验，数字记下来。
6. **再谈下一步**：第 5 步的命中率若不优于抛硬币（独立第三方实测是 50.2%，见速查表诚实标注区），就停在这里当研究玩具；明显优于基准，才轮得到换 base 档、加 sample_count、或按微调四步卡往下走。全程记住页首那行：模型输出不是投资建议。
