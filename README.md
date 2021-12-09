# Causal Imitative Model for Autonomous Driving

---

>  Mohammad Reza Samsami, Mohammadhossein Bahari, Saber Salehkaleybar, Alexandre Alahi.  arXiv 2021. <br /> 
>  __[[Project Website]](https://mrsamsami.github.io/causal-imitation/)__  &nbsp; &nbsp; &nbsp; __[[Paper]](https://arxiv.org/abs/2112.03908)__
> <img src="https://mrsamsami.github.io/causal-imitation/CIM/method.jpg" width="300"/>

---

This repo provides implementations of our work. All code is written in Python 3, using PyTorch, NumPy, and CARLA.

The project is built on [OATomobile](https://github.com/OATML/oatomobile), a research framework for autonomous driving. The main part of our contribution is gathered in ```oatomobile\baselines\torch\cim```.

## Installation
To install requirements, refer to [OATomobile](https://github.com/OATML/oatomobile) github repo.

## How to run

#### Train the perception model
To train the perception model, you would run with
```
python -m oatomobile.baselines.torch.cim.perception.train --dataset_dir=dataset_dir --output_dir=output_dir --in_channels=1 --num_epochs=num_epochs --beta=6
```

#### Train the speed predictor
After training the perception model and obtaining representations of scenarios' observations, you could train the speed predictor with
```
python -m oatomobile.baselines.torch.cim.predictor.train --dataset_dir=dataset_dir --output_dir=output_dir --num_epochs=num_epochs
```

#### Run a navigation task
To perform the model on a task: 
```
python -m test --task=task --model_dir=model_dir --predictor_dir=predictor_dir --output_dir=output_dir --alpha=alpha --gamma=gamma
```

## BibTeX
If you find this code useful, please cite:

```
@misc{samsami2021causal,
   title={Causal Imitative Model for Autonomous Driving}, 
   author={Mohammad Reza Samsami and Mohammadhossein Bahari and Saber Salehkaleybar and Alexandre Alahi},
   year={2021},
   eprint={2112.03908},
   archivePrefix={arXiv},
   primaryClass={cs.RO}
}
```

## Acknowledgements
This project is built on [OATomobile](https://github.com/OATML/oatomobile), a framework for autonomous driving research which wraps CARLA in OpenAI gym environments.
