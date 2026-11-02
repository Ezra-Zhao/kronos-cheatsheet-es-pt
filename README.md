# Kronos 速查表 | Kronos Cheatsheet | Guía rápida de Kronos

**模型输出不是投资建议、任何信号先过样本外与回测验证再谈**

> 一页看懂 Kronos：第一个专为金融 K 线训练的开源基础模型（清华系，论文已被 AAAI 2026 接收）。它先用一个专用 tokenizer 把 OHLCV 连续数值量化成层级离散 token，再用自回归 Transformer 像预测下一个词一样预测下一根 K 线，预训练数据来自 45 个以上全球交易所。K 线第一次有了自己的「GPT」——但它到底有没有用，本页把唱衰它的独立实测和原仓自己的免责声明一起摆出来，你自己判断。

- 上游项目：[shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)（2026-11-02 实测 45,851 ★ / 7,564 forks，MIT；论文 arXiv:2508.02739，已被 AAAI 2026 接收）
- 平台：Python 3.10+；模型与 tokenizer 在 Hugging Face（NeoQuasar）上，微调示例用 Qlib 备数据（以 A 股为例）
- 本页结构：中文主文 → English → Español → Português

---

Apoyo y más: [GitHub Sponsors](https://github.com/sponsors/Ezra-Zhao) · [Gumroad — plantillas y cuadernos que uso](https://ezraflow28.gumroad.com) · [Ko-fi](https://ko-fi.com/ezra777zhao)

## 它是什么、不是什么

| 是 | 不是 |
|---|---|
| 一个专读 K 线的基础模型本体：tokenizer + 自回归 Transformer 两段式，输出未来一段的 OHLCV 预测序列 | 不是一队 agent 开会投研（那是 TradingAgents 那类产品），这里没有观点、没有新闻、只有 K 线本身 |
| 开源可自托管：mini/small/base 三档已开源，在 Hugging Face 上直接下 | 不是买入信号发生器：原仓微调管线自带声明——只是演示，不是生产级量化系统（见诚实标注区） |
| 可以微调到自己的市场和周期上（原仓给了完整的四步管线示例） | 跟巴西散户搜到的付费交易指标「Sistema Kronos」**没有任何关系**——那是借同名卖指标的营销产品，本模型免费开源、不卖信号 |
| 论文已被 AAAI 2026 接收、arXiv 公开（2508.02739） | 学术接收不等于赚钱能力：独立样本外实测的方向命中率约等于抛硬币（证据见诚实标注区） |

## 模型四档选型表

| 档位 | 配套 tokenizer | 上下文长度 | 参数量 | 开源 |
|---|---|---|---|---|
| Kronos-mini | Kronos-Tokenizer-2k | 2048 | 4.1M | 已开源（NeoQuasar/Kronos-mini） |
| Kronos-small | Kronos-Tokenizer-base | 512 | 24.7M | 已开源（NeoQuasar/Kronos-small） |
| Kronos-base | Kronos-Tokenizer-base | 512 | 102.3M | 已开源（NeoQuasar/Kronos-base） |
| Kronos-large | Kronos-Tokenizer-base | 512 | 499.2M | **未开源** |

选型要点：tokenizer 不能配错——mini 用 2k 版，其余三档用 base 版；small/base 的上下文只有 512，这是后面出预测时最容易踩的坑。先拿 small 在自己数据上跑通，再决定要不要上 base。

## 五步出预测卡

1. **装依赖**：Python 3.10+，克隆原仓后 `pip install -r requirements.txt`。
2. **从 Hugging Face 载 tokenizer 与模型**：`KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")` 配 `Kronos.from_pretrained("NeoQuasar/Kronos-small")`（mini 则换 2k tokenizer 与 mini 模型，两边必须配套）。
3. **建 Predictor**：`KronosPredictor(model, tokenizer, max_context=512)`——预处理、归一化、预测、反归一化都由它包办。
4. **备齐三样输入**：`df`（历史 K 线 DataFrame，`open/high/low/close` 四列必备，`volume/amount` 可选、缺了补零）、`x_timestamp`（与 df 对应的历史时间戳序列）、`y_timestamp`（要预测的未来时间戳序列）。
5. **调 predict**：`predictor.predict(df, x_timestamp, y_timestamp, pred_len, T=1.0, top_p=0.9, sample_count=1)`，返回未来各期的 open/high/low/close/volume/amount；多条序列并行用 `predict_batch`（要求各序列 lookback 与 pred_len 一致）。

**坑先说清**：small 与 base 的 `max_context` 只有 512，输入长度（lookback）不要超过 512——超过 Predictor 会自动截断，但效果打折；原仓示例用 lookback 400、pred_len 120。`T`（温度）、`top_p`、`sample_count` 控制概率采样，多采样取平均出的是「多条路径的平均」，不是置信区间承诺。

## 微调四步卡（原仓以 A 股为例）

前置：装好 requirements，再 `pip install pyqlib` 并按 Qlib 官方指南把本地数据备好（示例用日频数据）。

1. **改配置**：所有路径与超参集中在 `finetune/config.py`——`qlib_data_path`、`dataset_path`、`save_path`、`backtest_result_path`、预训练 tokenizer/predictor 路径，以及 instrument、训练时段、epochs、batch_size；不用 Comet.ml 就把 `use_comet` 设为 False。
2. **备数据**：跑 `python finetune/qlib_data_preprocess.py`，在 dataset_path 下产出 train/val/test 三个 pkl。
3. **两段微调，顺序不能反**：先微调 tokenizer（`torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_tokenizer.py`，让它先适配你这个市场的数据分布；两个训练脚本都是 torchrun 多 GPU 口径，NUM_GPUS 填你的卡数），再微调 predictor（同法跑 `finetune/train_predictor.py`）。
4. **回测**：`python finetune/qlib_test.py --device cuda:0`，在测试集上生成信号，跑一个简单的 top-K 策略回测，输出绩效分析与累计收益曲线图（与基准对比）。

提醒：原仓 `finetune/` 目录里大量代码注释是 AI（Gemini 2.5 Pro）生成的，原仓明示可能有不准确处——以代码本身为准，别只读注释。

## 诚实标注区（这一栏是这张速查的命）

- **原仓自己的免责声明**：微调管线「intended as a demonstration」——是演示流程的简化示例，不是生产级量化交易系统；原仓同处还写明：demo 产出的是原始信号，真实量化流程要再过组合优化与风险因子中和；简单 top-K 回测也没有认真建模交易成本、滑点与市场冲击。
- **独立样本外实测唱衰它**（证据 B，单一第三方小样本）：一篇第三方实测在 25 个外汇与 CFD 市场上做了 749 次预测，方向命中率 50.2%——不优于抛硬币；按其信号交易，在计入成本之前就已经亏钱。样本不大、市场限于外汇与 CFD，但这是目前能找到的独立样本外数字，先摆在这里，别只看原仓的演示图。
- **页首那行再说一遍**：模型输出不是投资建议。任何拿它产出的信号，先过样本外、再过含成本的回测、再和简单基准（比如买入持有、抛硬币）对照，三关都过了才轮得到谈仓位。
- **多语文档现状**：原仓根目录只有英文 README；正文头部挂的西/葡语链接是 zdoc.app 的站外机器翻译，不是原仓文档、更不是速查——本页中文主文与西/葡译稿即为补这个空缺（译稿为学习译稿、待母语校对）。
- **势头数据**：45,851 ★ / 7,564 forks（2026-11-02 原仓页实测，证据 A），fork 率约 16%——是真有人拿回去跑自己数据的信号；今日仍在第三方今日榜上（约 +630 星口径，证据 B）。

---

## English (summary)

Kronos (45,851 stars / 7,564 forks as of 2026-11-02, MIT; paper arXiv:2508.02739, accepted to AAAI 2026) is the first open-source foundation model trained specifically on financial candlesticks, pre-trained on data from 45+ global exchanges. A specialized tokenizer first quantizes continuous OHLCV values into hierarchical discrete tokens; an autoregressive Transformer then predicts the next K-line the way a language model predicts the next word. **Model output is not investment advice — any signal must pass out-of-sample and backtest validation before it is even discussed.**

Four sizes: mini (4.1M params, context 2048, pairs with Kronos-Tokenizer-2k), small (24.7M, context 512, Tokenizer-base), base (102.3M, context 512, Tokenizer-base) — all open on Hugging Face — and large (499.2M, context 512), which is **not** open-sourced. Forecasting takes five steps: install dependencies (Python 3.10+, `pip install -r requirements.txt`); load the matching tokenizer and model from Hugging Face; build a `KronosPredictor` (max_context 512 for small/base); prepare the three inputs — a `df` with at least open/high/low/close, the historical `x_timestamp` series and the future `y_timestamp` series; then call `predict` (T / top_p / sample_count control probabilistic sampling). Watch the pitfall: for small and base your lookback should not exceed 512. Fine-tuning (the repo's example uses Qlib on Chinese A-shares) runs config → data preparation → fine-tune the tokenizer first, then the predictor → backtest with a simple top-K strategy.

Honest notes: the upstream repo's own disclaimer says the fine-tuning pipeline is a demonstration, not a production-ready quantitative trading system, and its raw signals still need portfolio optimization, risk-factor neutralization and realistic cost modeling. An independent out-of-sample test (one third-party study, evidence level B: 25 forex/CFD markets, 749 forecasts) measured a 50.2% directional hit rate — no better than a coin flip — and trading its signals lost money even before costs. The Spanish/Portuguese links in the upstream README are third-party machine translations (zdoc.app), not repo documentation. And the paid trading indicator marketed in Portuguese as "Sistema Kronos" has nothing to do with this free, open-source model.

---

## Español (guía rápida — traducción de estudio, pendiente de revisión por un hablante nativo)

**La salida del modelo no es un consejo de inversión: cualquier señal debe pasar validación fuera de muestra y backtesting antes de hablar de ella.**

Kronos (45.851 estrellas / 7.564 forks a 2 de noviembre de 2026, MIT; artículo arXiv:2508.02739, aceptado en AAAI 2026) es el primer modelo fundacional de código abierto entrenado específicamente con velas financieras (K-lines), preentrenado con datos de más de 45 bolsas de todo el mundo. Un tokenizador especializado cuantiza primero los valores continuos OHLCV en tokens discretos jerárquicos; después, un Transformer autorregresivo predice la siguiente vela como un modelo de lenguaje predice la siguiente palabra.

Cuatro tamaños: mini (4,1 M de parámetros, contexto 2048, con Kronos-Tokenizer-2k), small (24,7 M, contexto 512, Tokenizer-base), base (102,3 M, contexto 512, Tokenizer-base) — los tres abiertos en Hugging Face — y large (499,2 M, contexto 512), que **no** es de código abierto. Predecir lleva cinco pasos: instalar dependencias (Python 3.10+, `pip install -r requirements.txt`); cargar desde Hugging Face el tokenizador y el modelo que hacen pareja; crear un `KronosPredictor` (max_context 512 en small/base); preparar las tres entradas — un `df` con al menos open/high/low/close, la serie histórica `x_timestamp` y la serie futura `y_timestamp`; y llamar a `predict` (T / top_p / sample_count controlan el muestreo probabilístico). El tropiezo típico: en small y base, tu lookback no debería superar 512. El ajuste fino (el ejemplo del repositorio usa Qlib con acciones A de China) va: configuración → preparación de datos → ajustar primero el tokenizador y después el predictor → backtesting con una estrategia top-K sencilla.

**Notas honestas**: el propio repositorio advierte que su pipeline de ajuste fino es una demostración, no un sistema de trading cuantitativo listo para producción, y que sus señales en bruto aún necesitan optimización de cartera, neutralización de factores de riesgo y un modelado realista de costes. Una prueba independiente fuera de muestra (un solo estudio de terceros, evidencia nivel B: 25 mercados de forex y CFD, 749 predicciones) midió una tasa de acierto direccional del 50,2 % — no mejor que lanzar una moneda — y operar sus señales perdió dinero incluso antes de costes. Los enlaces en español/portugués del README original son traducciones automáticas de terceros (zdoc.app), no documentación del repositorio.

---

## Português (guia rápido — tradução de estudo, aguardando revisão de um falante nativo)

**A saída do modelo não é recomendação de investimento: qualquer sinal precisa passar por validação fora da amostra e backtest antes de ser levado a sério.**

Kronos (45.851 estrelas / 7.564 forks em 2 de novembro de 2026, MIT; artigo arXiv:2508.02739, aceito na AAAI 2026) é o primeiro modelo fundacional de código aberto treinado especificamente em candles financeiros (K-lines), pré-treinado com dados de mais de 45 bolsas do mundo. Um tokenizador especializado primeiro quantiza os valores contínuos OHLCV em tokens discretos hierárquicos; depois, um Transformer autorregressivo prevê o próximo candle como um modelo de linguagem prevê a próxima palavra.

Quatro tamanhos: mini (4,1 M de parâmetros, contexto 2048, com Kronos-Tokenizer-2k), small (24,7 M, contexto 512, Tokenizer-base), base (102,3 M, contexto 512, Tokenizer-base) — os três abertos no Hugging Face — e large (499,2 M, contexto 512), que **não** é de código aberto. Prever leva cinco passos: instalar as dependências (Python 3.10+, `pip install -r requirements.txt`); carregar do Hugging Face o tokenizador e o modelo que formam o par; criar um `KronosPredictor` (max_context 512 no small/base); preparar as três entradas — um `df` com pelo menos open/high/low/close, a série histórica `x_timestamp` e a série futura `y_timestamp`; e chamar `predict` (T / top_p / sample_count controlam a amostragem probabilística). A armadilha típica: no small e no base, seu lookback não deve passar de 512. O ajuste fino (o exemplo do repositório usa Qlib com ações A da China) segue: configuração → preparação de dados → ajustar primeiro o tokenizador e depois o predictor → backtest com uma estratégia top-K simples.

**Notas honestas**: o próprio repositório avisa que seu pipeline de ajuste fino é uma demonstração, não um sistema de trading quantitativo pronto para produção, e que seus sinais brutos ainda precisam de otimização de carteira, neutralização de fatores de risco e uma modelagem realista de custos. Um teste independente fora da amostra (um único estudo de terceiros, evidência nível B: 25 mercados de forex e CFD, 749 previsões) mediu uma taxa de acerto direcional de 50,2 % — não melhor que cara ou coroa — e operar seus sinais perdeu dinheiro mesmo antes dos custos. Os links em espanhol/português do README original são traduções automáticas de terceiros (zdoc.app), não documentação do repositório. E o indicador de trading pago vendido em português como «Sistema Kronos» não tem nenhuma relação com este modelo gratuito e de código aberto.

---

## 来源与声明

- 灵感来源 / Source / Fuente / Fonte: [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)（论文：arXiv:2508.02739，AAAI 2026；在线演示 shiyu-coder.github.io/Kronos-demo）
- 本仓是个人学习整理：内容用我自己的话总结自上游公开文档，不搬运原文；模型档位、调用步骤与微调顺序以 2026-11-02 的上游 README（Model Zoo 与 Getting Started）为准，上游迭代后请以官方文档为最终依据。诚实标注区的第三方实测为单一来源（证据 B），引用时请保留其样本口径。
- 本页不构成任何投资建议。Este contenido no es un consejo de inversión. Este conteúdo não é recomendação de investimento.
- 西语、葡语部分为学习译稿，待母语校对。Las secciones en español y portugués son traducciones de estudio pendientes de revisión nativa.
- License: MIT © 2026 Guangyi Zhao
