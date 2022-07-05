# bert-ruber-kor-pytorch
This repository is the Korean implementation of **BERT-RUBER: Better Automatic Evaluation of Open-Domain Dialogue Systems with Contextualized Embeddings**[[1]](#1).

The project includes sample dialogue data, training/evaluation code, and a BERT-RUBER class module for actual inference.

The data which is used here is from "한국어 감정 정보가 포함된 연속적 대화 데이터셋"[[2]](#2), which can be downloaded from [here](https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-010). (You might need additional registration & application for download.)

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

**Arguments for data parsing**

| Argument        | Type    | Description                                | Default               |
| --------------- | ------- | ------------------------------------------ | --------------------- |
| `seed`          | `int`   | The random seed.                           | `555`                 |
| `raw_data_path` | `str`   | The raw xlsx data file path.               | *YOU SHOULD SPECIFY.* |
| `train_ratio`   | `float` | The ratio of train set to total data size. | `0.9`                 |

<br/>

**Arguments for training**

| Argument           | Type    | Description                                                  | Default               |
| ------------------ | ------- | ------------------------------------------------------------ | --------------------- |
| `seed`             | `int`   | The random seed.                                             | `555`                 |
| `model_path`       | `str`   | The pre-trained BERT checkpoint to fine-tune.                | *YOU SHOULD SPECIFY.* |
| `default_root_dir` | `str`   | The default directory for logs & checkpoints.                | `"."`                 |
| `data_dir`         | `str`   | The directory which contains data files.                     | `data`                |
| `max_len`          | `int`   | The maximum length of each input.                            | `256`                 |
| `num_epochs`       | `int`   | The number of total epochs.                                  | `5`                   |
| `train_batch_size` | `int`   | The batch size for training.                                 | `32`                  |
| `eval_batch_size`  | `int`   | The batch size for evaluation.                               | `16`                  |
| `num_workers`      | `int`   | The number of workers for data loading.                      | `4`                   |
| `warmup_ratio`     | `float` | The ratio of warmup steps to total training steps.           | `0.1`                 |
| `max_grad_norm`    | `float` | The maximum value for gradient clipping.                     | `1.0`                 |
| `learning_rate`    | `float` | The initial learning rate.                                   | `5e-5`                |
| `gpus`             | `str`   | The indices of GPUs to use. To use multiple GPUs, use commas. (e.g. `"0, 1, 2, 3"`) | `"0"`                 |
| `pooling`          | `str`   | The pooling method.                                          | *YOU SHOULD SPECIFY.* |
| `w1_size`          | `int`   | The size of w1 embedding.                                    | `768`                 |
| `w2_size`          | `int`   | The size of w2 embedding.                                    | `256`                 |
| `w3_size`          | `int`   | The size of w3 embedding.                                    | `64`                  |
| `num_hists`        | `int`   | The number of extra histories.                               | `0`                   |

<br/>

**Arguments for extracting a checkpoint**

| Argument           | Type  | Description                                   | Default               |
| ------------------ | ----- | --------------------------------------------- | --------------------- |
| `default_root_dir` | `str` | The default directory for logs & checkpoints. | `"."`                 |
| `log_idx`          | `int` | The lightning log index.                      | *YOU SHOULD SPECIFY.* |
| `ckpt_file`        | `str` | The checkpoint file name to extract.          | *YOU SHOULD SPECIFY.* |

<br/>

---

### How to run

1. Install all required packages.

   ```shell
   pip install -r requirements.txt
   ```

   <br/>

2. Parse the raw data into json format.

   ```shell
   sh exec_parse_data.sh
   ```

   If you don't need to use the default data, prepare your data files as follows.

   ```
   data (default name of data directory)
   └--train_dials.json
   └--eval_dials.json
   ```

   And each data file contains dialogues in a list form.

   ```python
   # For example...
   [
     [ # dialogue 1
       "오늘 뭐 먹었니?",
       "나는 치킨을 먹었어.",
       "어디서 시켜먹었어?",
       "우리 집 앞 치킨 집에서!"
     ],
     [  # dialogue 2
       "날씨가 정말 좋다!",
       "등산이라도 갈까?"
     ],
     ...
   ]
   ```

   <br/>

3. Train a BERT-RUBER model using a pre-trained BERT checkpoint.

   ```shell
   sh exec_train.sh
   ```

   <br/>

4. Extract the fine-tuned checkpoint for later usage.

   ```shell
   sh exec_extract_ckpt.sh
   ```

<br/>

---

### Results

<br/>

---

### Inference

<br/>

---

### References

<a id="1">[1]</a> Ghazarian, S., Wei, J. T. Z., Galstyan, A., & Peng, N. (2019). Better automatic evaluation of open-domain dialogue systems with contextualized embeddings. *arXiv preprint arXiv:1904.10635*. <a href="https://arxiv.org/pdf/1904.10635.pdf">https://arxiv.org/pdf/1904.10635.pdf</a>

<a id="2">[2]</a> 한국어 감정 정보가 포함된 연속적 대화 데이터셋. <a href="https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-010">https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-010</a>

<a id="3">[3]</a> Tao, C., Mou, L., Zhao, D., & Yan, R. (2018, April). Ruber: An unsupervised method for automatic evaluation of open-domain dialog systems. In *Thirty-Second AAAI Conference on Artificial Intelligence*. <a href="https://arxiv.org/pdf/1701.03079.pdf">https://arxiv.org/pdf/1701.03079.pdf</a>
