# bert-ruber-kor-pytorch
This repository is the Korean implementation of **BERT-RUBER: Better Automatic Evaluation of Open-Domain Dialogue Systems with Contextualized Embeddings**[[1]](#1).

The project includes sample dialogue data, training/evaluation code, and a BERT-RUBER class module for actual inference.

The data which is used here is from XPersona[[2]](#2), which can be downloaded from [here](https://github.com/HLTCHKUST/Xpersona).

<br/>

---

### Data pre-processing

The original RUBER[[3\]](#3) and BERT-RUBER consider each input as a single-turn utterance, which is difficult to apply to most human dialogues, which are continuous.

Also, extra information, such as persona descriptions, can be useful for evaluating the quality of system responses.

Therefore, I implemented not only the data preprocessing method in the original paper but also improved one with further contexts, which are previous dialogue history and persona sentences.

<br/>

The detailed descriptions are as follows.

![1](https://user-images.githubusercontent.com/16731987/176834729-a0f2f48e-e627-4e24-834f-92eb4e905662.png)

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

<a id="2">[2]</a> Lin, Z., Liu, Z., Winata, G. I., Cahyawijaya, S., Madotto, A., Bang, Y., ... & Fung, P. (2020). XPersona: Evaluating multilingual personalized chatbot. *arXiv preprint arXiv:2003.07568*. <a href="https://aclanthology.org/2021.nlp4convai-1.10.pdf">https://aclanthology.org/2021.nlp4convai-1.10.pdf</a>

<a id="3">[3]</a> Tao, C., Mou, L., Zhao, D., & Yan, R. (2018, April). Ruber: An unsupervised method for automatic evaluation of open-domain dialog systems. In *Thirty-Second AAAI Conference on Artificial Intelligence*. <a href="https://arxiv.org/pdf/1701.03079.pdf">https://arxiv.org/pdf/1701.03079.pdf</a>
