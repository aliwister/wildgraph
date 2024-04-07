# WildGraph: Realistic Graph-based Trajectory Generation for Wildlife
This repository contains the official implementation of [WildGraph: Realistic Graph-based Trajectory Generation for Wildlife]() submitted to KDD'24.

![](assets/wilgraph.pdf)




## Gettting Started

1\. Clone this repository:
```
git clone [https://github.com/aliwister/wildgraph.git](https://github.com/aliwister/wildgraph.git)
cd wildgraph
```

2\. Create a conda environment and install the dependencies:
```
conda create --name wildgraph --file requirements.txt
```

## Training

It is quite easy to train and test WildGraph or any of the benchmark methods reported:

<code>python wild_run.py --dataset <geese|stork> --exp <WILDGRAPH|GAN|VAE|WILDGEN|TRANSFORMER> --epochs <epochs> --split_distance <r> --num_exps <number of experiments to average> --desc <a general description></code>


To train WildGraph:
```
python wild_run.py --dataset geese --exp WILDGRAPH --epochs 100 --split_distance .25


```

To train VAE:
```
python wild_run.py --dataset geese --exp VAE --epochs 100 
```

After training, a report will be saved in `wild_experiments_log/[EXP]` automatically.


## Citation

If you found this repository useful, please consider citing our work:

```
@inproceedings{
}
```

## License

This repository is licensed under [Apache 2.0](LICENSE).
