# Eq.(5) 描述替换稿

> 历史阶段报告：正文保留原阶段范围与结果；文内证据链接固定到原始提交391d7e2，不指向当前120条派生数据。

用户已确认保留现有 `CandidateRecord.semantic_score`。下文按冻结实现整理为可供论文采用的文字；当前任务资料中未找到论文 TeX 源文件，因此本文件是替换稿，尚未应用到论文主文件。

## English text

For a candidate at location $x_n$, let $g_n$ denote the selected Gaussian. We use its weighted semantic codebook score as the evidence for target category $k$:

$$
E_n = \sum_{j=1}^{J_n} w_{n,j}\,\mathbf f_{I_{n,j}}^{\mathsf T}\widehat{\mathbf t}_k,
\qquad
\widehat{\mathbf t}_k = \frac{\mathbf t_k}{\lVert\mathbf t_k\rVert_2}.
$$

Here, $I_{n,j}$ and $w_{n,j}$ are the stored codebook indices and weights of $g_n$, $\mathbf f_{I_{n,j}}$ is the corresponding codebook feature, and $\widehat{\mathbf t}_k$ is the normalized target text embedding. The candidate record stores this value as its semantic score. Spatial support statistics govern candidate filtering and optional support reranking; the semantic score supplied to the belief model remains the score of $g_n$.

## LaTeX

```tex
For a candidate at location $x_n$, let $g_n$ denote the selected Gaussian.
We use its weighted semantic codebook score as the evidence for target
category $k$:
\begin{equation}
E_n = \sum_{j=1}^{J_n} w_{n,j}\,
      \mathbf f_{I_{n,j}}^{\mathsf T}\widehat{\mathbf t}_k,
\qquad
\widehat{\mathbf t}_k = \frac{\mathbf t_k}{\lVert\mathbf t_k\rVert_2}.
\label{eq:semantic_evidence}
\end{equation}
Here, $I_{n,j}$ and $w_{n,j}$ are the stored codebook indices and weights
of $g_n$, $\mathbf f_{I_{n,j}}$ is the corresponding codebook feature,
and $\widehat{\mathbf t}_k$ is the normalized target text embedding.
The candidate record stores this value as its semantic score.
Spatial support statistics govern candidate filtering and optional
support reranking; the semantic score supplied to the belief model
remains the score of $g_n$.
```

## 与实现的对应

- [查询分数](https://github.com/pieceHk/ForesightNav/blob/391d7e25b12171144a77cfbc41f220efe11294fb/pbsp/query/engine.py#L94)：`raw = feat_dict @ text_feature.T`，随后按 `ok_idx` 取出各高斯的码本项，以 `ok_wts` 加权求和。
- [候选记录](https://github.com/pieceHk/ForesightNav/blob/391d7e25b12171144a77cfbc41f220efe11294fb/pbsp/query/engine.py#L248)：目标文本向量先做 L2 归一化，`diagnostics['selected_scores']` 写入 `semantic_score`；support count、support score 和 combined rerank score 分列保存。
- [BLP 输入](https://github.com/pieceHk/ForesightNav/blob/391d7e25b12171144a77cfbc41f220efe11294fb/pbsp/navigation/blp.py#L175)：直接取 `record.semantic_score` 作为 `E`。

因此应删除原文中“支撑高斯集合上的平均目标相关度”这一解释。这里没有对邻近支撑集的语义分数取空间平均，也不额外假设码本特征为单位向量或权重之和为1；上述表达仅对应当前加权点积实现。

核对基准为运行提交 `03242fda76fd6a631596a7269c8e7990c657ea30` 及 CP2b 接受的161项运行输入。`pbsp/query/engine.py` 的 SHA256 为 `2a7bf07ac1ffea1797c33b45a708c3ce9d8bef94d5581642370e64acd85c7f57`，`pbsp/navigation/blp.py` 为 `fedf1b03677d45e4ce0860af85b86f6a05faf024489250d2c437b59f25a2cd50`。本次只准备论文文字，没有改动 E、覆盖度、标定数值、候选排序或已有接受记录。S 和 belief 的具体定义继续沿用既有实现与说明。
