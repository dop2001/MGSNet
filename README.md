# What is MGSNet?

Myasthenia gravis (MG) is an autoimmune disease characterized by chronic neuromuscular transmission disorders, causing varying degrees of muscle weakness. It most commonly affects the eyes, face, and muscles involved in swallowing. It can cause symptoms such as double vision, drooping eyelids, difficulty speaking and walking. MGSNet is an efficient and simple model framework for detecting MG. By collecting voice data, MGSNet can be used to quickly diagnose whether you have MG.



# File structure

```bash
.
├── README.md
├── datasets	# Store dataset
├── logs	# Save training and testing logs
├── models	# MGSNet
├── pths	# Save training weights
├── runs	# Tensorboard
├── train.py	# Training model
└── utils	# Some tool files
    ├── dataloader.py
    ├── loger.py
    ├── machine_learning.py
    ├── metrics.py
    └── random_state.py
```



# Training

1.Clone the code and prepare the required environment.

```shell
git clone https://github.com/dop2001/MGSNet.git
cd MGSNet
pip install -r requirements.txt
```

2.Train and test MGSNet, and provide machine learning comparison models.

```shell
python train.py
# machine learning comparison models
python utils/machine_learning.py
```





