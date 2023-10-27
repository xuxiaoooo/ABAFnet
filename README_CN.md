**[English](README.md)** | **[中文](README_CN.md)**

# ABAFnet

<p align="center">
  <img src="https://github.com/xuxiaoooo/ABAFnet/blob/main/draw/LOGO 1.png" width="400" height="200" alt="logo"/>
</p>

<p align="center">
  基于注意力机制的声学特征融合网络用于抑郁症检测
</p>

<p align="center">
  <img src="https://github.com/xuxiaoooo/ABAFnet/blob/main/draw/fig2.jpg" width="600" height="600" alt="Backbone Flow"/>
</p>


---

## 📙 数据集申请

我们团队将致力于多人群多模态的大数据集采集以及人工智能方法的研究。

数据的公开申请将第一时间在我们团队的所有相关仓库获取（由于隐私策略，大部分暂未获得公开许可）（如果作为审稿人对文章有质疑，请联系通讯作者获取特征数据文件）。

音视频相关数据和研究请持续关注本代码仓库，脑影像、脑电、生理信号等数据请关注通讯作者 Xizhe Zhang 的后续论文发表。

---

## 📌 介绍

这个仓库包含了我们提出的 **ABAFnet** 的实现，这是一个新颖的基于注意力机制的语音特征融合循环网络，用于有效的抑郁症检测和分析。这个项目的主要目标是提供一个准确且高效的方式来使用语音数据进行抑郁症的检测。

_**注:** 我们发布的代码由于数据隐私问题，在某些细节上隐去了数据加载部分，因此并非是完整版本，有些部分是代码的调试版本，但是模型结构部分是最新的。由于我们对数据隐私的关注，我们隐藏了关于数据的部分，请通过电子邮件与我们联系如果你需要。_

---

## 💡 特性
- 从语音数据中提取特征。
- 注意力机制用于更好的特性表示。
- 循环网络架构用于模型时间信息。
- 高效且准确的抑郁症检测。

---

## 🛠️ 安装和使用
**克隆仓库**
```bash
git clone https://github.com/xuxiaoooo/ABAFnet.git
cd ABAFnet
```
---
## 📊 结果

我们提出的 ABAFnet 在基于语音的抑郁症检测中取得了最先进的表现。详细的结果和与其他方法的比较可以在我们的论文中找到。

---

## 📄 引用
文章链接:
Arxiv: https://arxiv.org/pdf/2308.12478v1.pdf


Research Gate: https://www.researchgate.net/publication/373364067_Attention-Based_Acoustic_Feature_Fusion_Network_for_Depression_Detection#fullTextFileContent

如果你发现这项工作有帮助，请引用我们的论文：
```
@misc{xu2023attentionbased,
      title={Attention-Based Acoustic Feature Fusion Network for Depression Detection}, 
      author={Xiao Xu and Yang Wang and Xinru Wei and Fei Wang and Xizhe Zhang},
      year={2023},
      eprint={2308.12478},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

---

## 📧 联系

有任何问题，欢迎开启一个 Issue或者通过 xuxiaooo1111@gmail.com 与我们联系。

---
