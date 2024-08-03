# KANSC

<center>Zifan Zhu, Computer Science of Western University</center>

KAN integration of Deep Learning Enabled Semantic Communication Systems

## DeepSC author:

<center>Huiqiang Xie, Zhijin Qin, Geoffrey Ye Li, and Biing-Hwang Juang </center>

## Requirements

- Refer to readme of [this](https://github.com/KindXiaoming/pykan) repository for setting up Pykan environment

- See the `requirements.txt` for the required python packages for original DeepSC and run `pip install -r requirements.txt` to install them.

## Preprocess

```shell
mkdir data
wget http://www.statmt.org/europarl/v7/europarl.tgz
tar zxvf europarl.tgz
python preprocess_text.py
```

## Train

- For training KANSC model

```shell
python main_kan.py
```

- For training DeepSC model

```shell
python main.py
```

## Evaluation

Check `kan_sc_performance.ipynb` file for evaluation of KANSC model
Check `performance.ipynb` file for evaluation of DeepSC model

## Notes

- The Integrated version only focuses on evaluating the BLEU score performance vs. different SNR over AWGN and Rayleigh Fading Channel

- The `efficient_kan` implementation of `KAN` layer is from [this](https://github.com/Blealtan/efficient-kan) repository

## Bibtex

```bitex
@article{xie2021deep,
  author={H. {Xie} and Z. {Qin} and G. Y. {Li} and B. -H. {Juang}},
  journal={IEEE Transactions on Signal Processing},
  title={Deep Learning Enabled Semantic Communication Systems},
  year={2021},
  volume={Early Access}}
```
