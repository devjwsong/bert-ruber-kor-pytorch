# bert-ruber-kor-pytorch
This repository is the Korean implementation of **BERT-RUBER: Better Automatic Evaluation of Open-Domain Dialogue Systems with Contextualized Embeddings**[[1]](#1).

The project includes sample dialogue data, training/evaluation code, and a BERT-RUBER class module for actual inference.

The data which is used here is from "한국어 감정 정보가 포함된 연속적 대화 데이터셋"[[2]](#2), which can be downloaded from [here](https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-010).

<br/>

---

### Data pre-processing

The original RUBER[[3\]](#3) and BERT-RUBER consider each input as a single-turn utterance, which is difficult to apply to most human dialogues, which are continuous.

Therefore, I implemented not only the data preprocessing method in the original paper but also improved one with further contexts, which are previous dialogue histories.

<br/>

The detailed descriptions are as follows.

![1](https://user-images.githubusercontent.com/16731987/177268507-106f5c85-dba3-4f0b-8718-4b03af419dae.png)

<br/>

You can skip the data parsing script and use your own dataset to train your models.

The details of how to prepare your data will be introduced in the later section.

<br/>

---

### Arguments

<br/>

---

### How to run

<br/>

---

### References

<a id="1">[1]</a> Ghazarian, S., Wei, J. T. Z., Galstyan, A., & Peng, N. (2019). Better automatic evaluation of open-domain dialogue systems with contextualized embeddings. *arXiv preprint arXiv:1904.10635*. <a href="https://arxiv.org/pdf/1904.10635.pdf">https://arxiv.org/pdf/1904.10635.pdf</a>

<a id="2">[2]</a> 한국어 감정 정보가 포함된 연속적 대화 데이터셋. <a href="https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-010">https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-010</a>

<a id="3">[3]</a> Tao, C., Mou, L., Zhao, D., & Yan, R. (2018, April). Ruber: An unsupervised method for automatic evaluation of open-domain dialog systems. In *Thirty-Second AAAI Conference on Artificial Intelligence*. <a href="https://arxiv.org/pdf/1701.03079.pdf">https://arxiv.org/pdf/1701.03079.pdf</a>
