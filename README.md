# Traffic Prediction with a graph-based U-Net architecture

This repo contains code to train and evaluate models to predict traffic volue and speed on the data of the *Traffic4Cast* challenge [[1, 2]](#T4C1).
[[PDF Version of the Paper]](https://ieeexplore.ieee.org/document/9892453)
[[PDF Version of the Slides]](https://github.com/LucaHermes/graph-unet-traffic-prediction/files/7726924/Traffic4Cast.-.Hybrid.Graph.U-Net.pdf)

![demo_six_cities_2019-04-24_frame156_traffic_sum-1](https://user-images.githubusercontent.com/30961397/139871911-767448ab-2fa5-4b16-8ba8-9c397a5d0e26.png)
*Sum over 24 consecutive frames of the traffic movies for six different cities on 24.04.2019 starting at 13:00. The color values are scaled logarithmically for better visibility. [PDF Version](https://github.com/LucaHermes/graph-unet-traffic-prediction/files/7461617/demo_six_cities_2019-04-24_frame156_traffic_sum.pdf)*

## Abstract

The IARAI *Traffic4Cast* challenge aims at predicting future traffic in the scale of whole cities. The competitions held in past years have shown that U-Net models perform well on this task. In this work, we point out the advantages of applying graph neural networks instead of visual convolutions and propose a U-Net-like model with graph layers. We further specialize existing graph operations to be sensitive to geographical topology and generalize pooling and upsampling operations to be applicable to graphs.
We finally show that our model generalizes well to unseen cities.

## Core Concept

As deep convolutional models encode the full input image, later layers will learn patterns specific to the cities in the training set. Intuitively, this limits spatial generalization capabilities. To remove this limitation, we replace the convolutions in UNet with graph layers. The given graph can be described as an incomplete pixel grid. Thus, when comparing two nodes ```A``` and ```B```, we can assign a geographic relationship between ```A``` and ```B```. For example, node ```A``` may be positioned to the south of node ```B```. As edges only ever connect adjacent nodes, there are eight possible values that this relationshoip can have: north (N), north-east (NE), east (E), south-east (SE), south (S), south-west (SW), west (W), north-west (NW).
There are two options of incorporating this information in graph models. First, by using edge feature-vectors that specify this geographic relation. Second, by adapting the neighborhood accumulation function in graph layers. We choose the second option and use different learnable transformations depending on the relative position of the neighboring node. Instead of differentiating between all eight, we only differentiate the quadrant NE, SE, SW, NW, as this reflects the way the traffic information is encoded and also reduces the number of transformations. This is tecnically realized by subdividing the graph into four subgraphs that only contain the respective edge direction.

![road_subgraphs-1](https://user-images.githubusercontent.com/30961397/140372978-2325f91a-93b1-421e-9bf0-b1228d750832.png)
*The four directed subgraphs that we extracted from the road graph. The graphs exclusively contain edges in the respective geographical direction.*

Subgraph creation is done in [```data/data_utils```](https://github.com/LucaHermes/graph-unet-traffic-prediction/blob/main/data/data_utils.py#L47-L58). The graph operation is defined in [```layers/hybrid_gnn.py```](https://github.com/LucaHermes/graph-unet-traffic-prediction/blob/main/layers/hybrid_gnn.py#L10). As you can see, we iterate over the subgraphs (s. [```line 85```](https://github.com/LucaHermes/graph-unet-traffic-prediction/blob/main/layers/hybrid_gnn.py#L85)) and apply separate dense layers in every iteration. This ultimately leads to output node features that are sensitive to the geographical neighborhood topology.

## What's contained in this repository
 
 * Our proposed Graph-UNet architecture together with a [checkpoint](https://github.com/LucaHermes/graph-unet-traffic-prediction/tree/main/ckpts/GraphUNet/GraphUNet_03-10-2021__16-04-37) with pretrained parameters
     * [```Code```](https://github.com/LucaHermes/graph-unet-traffic-prediction/blob/main/models/graph_unet.py) The coarse-grade architecture is inspired by the UNet [[3]](#UNet)
     * Each downsampling block consists of 
         * [```Code```](https://github.com/LucaHermes/graph-unet-traffic-prediction/blob/main/layers/hybrid_gnn.py#L10-L137) A modified version of the *full GN Block* as presented in [[4]](#fullGNBlock) 
         * [```Code```](https://github.com/LucaHermes/graph-unet-traffic-prediction/blob/main/layers/hybrid_gnn.py#L410-L447) And a custom unparameterized downsampling operation thats a generalization from the visual case to the given grid-like graph  
     * Each upsampling block consists of 
         * [```Code```](https://github.com/LucaHermes/graph-unet-traffic-prediction/blob/main/layers/hybrid_gnn.py#L254-L389) A custom graph upsampling layer  
         * Two consecutive layers of the modified *full GN Block*
 * [```Code```](https://github.com/LucaHermes/graph-unet-traffic-prediction/blob/main/models/unet.py) An implementation of the vanilla UNet model [[3]](#UNet)
     * We use nearest-neighbor upsampling instead of transposed convolution
     * No batch normalization is applied

## Setup and Usage of the Code

### Get the data and setup the environment

 1. Download the dataset from the competition (unfortunately the link cannot be published here).
 2. Extract the data anywhere you like
 3. Install the dependencies: ```pip install -r requirements.txt``` (we use Python 3.8.10)

### Train a model

 * We provide a CLI to train and test the models in [```train.py```](https://github.com/LucaHermes/graph-unet-traffic-prediction/blob/main/train.py#L111-L163)
 * We use [```config.py```](https://github.com/LucaHermes/graph-unet-traffic-prediction/blob/main/config.py) to set a default configuration of the model which can be altered via the CLI
 * General usage:
```
  --model 
      : Select the model to train (s. config.py for a list of possible models to train)
  --activation, --depth, --units, --use_bias
      : Set model hyperparameters with these arguments
  --layer_type
      : (Only graph-models) Used to specify a layer type for graph-models
  --batch, --epochs,
      : Configure the training loop
  --acc_gradients_steps 
      : Set the number of gradient accumulation steps before updating the model
  --data_type, --data_dir 
      : Configure the dataset
```
 
 * Besides model hyperparameters, three options are especially important: 
     * ```data_type``` specifies wheather the data should be loaded as graphs or as images
     * ```data_dir``` specifies where the dataset is located on your hard drive
     * ```model``` specifies the model to train

#### Examples
Train a vanilla UNet model with eight down- and upsampling blocks, respectively. Note that the data in this case is located in ```data/raw```:
```
python train.py --model UNet \
             --depth 8 --units 8 \
             --batch 2 --epochs 15 --acc_gradients_steps 8 \
             --data_type 'image' --data_dir data/raw 
```

Train a ReLU-activated Graph-UNet model from the provided checkpoint:
```
python train.py --model GraphUNet \
             --batch 1 --epochs 15 --acc_gradients_steps 16 \
             --data_type 'graph' --data_dir data/raw \
             --checkpoint ckpts/GraphUNet/GraphUNet_03-10-2021__16-04-37
```
> When first training a model that uses graph data, all dataset files are preprocessed to speed up the data pipeline

### Create a submission

 * The code to generate submissions is contained in [```submission.py```](https://github.com/LucaHermes/graph-unet-traffic-prediction/blob/main/submission.py)
 * It uses the same CLI as [```train.py```](https://github.com/LucaHermes/graph-unet-traffic-prediction/blob/main/train.py)
 * We only ever submitted the test results from the GraphUNet, so the submission script was made only for that model.
#### Example
```
python submission.py --model GraphUNet \
             --batch 1 --layer_type 'geo_quadrant_gcn' \
             --data_type 'graph' --data_dir data/raw \
             --checkpoint ckpts/GraphUNet/GraphUNet_03-10-2021__16-04-37/
```

### Reproduce the Results

Here, ```n_days``` is set to 3 to make the script run quicker. In our evaluations, we set ```n_days``` to 30 to reduce the variance in the results.
The figures will be saved in ```results/<model_id>```.
```  
python results.py --model GraphUNet \
             --batch 1 --layer_type 'geo_quadrant_gcn' \
             --data_type 'graph' --data_dir data/raw \
             --checkpoint ckpts/GraphUNet/GraphUNet_03-10-2021__16-04-37/ \
             --n_days 3
```

## Citation
```
@INPROCEEDINGS{hermes_graphunet_2022,
  author={Hermes, Luca and Hammer, Barbara and Melnik, Andrew and Velioglu, Riza and Vieth, Markus and Schilling, Malte},
  booktitle={2022 International Joint Conference on Neural Networks (IJCNN)},
  title={A Graph-based U-Net Model for Predicting Traffic in unseen Cities},
  year={2022},
  pages={1-8},
  doi={10.1109/IJCNN55064.2022.9892453}
}
```

## References
 * <a id="T4C1">[1]</a> Kopp, M., Kreil, D., Neun, M., Jonietz, D., Martin, H., Herruzo, P., Gruca, A., Soleymani, A., Wu, F., Liu, Y., Xu, J., Zhang, J., Santokhi, J., Bojesomo, A., Marzouqi, H., Liatsis, P., Kwok, P., Qi, Q., & Hochreiter, S. (2021). Traffic4cast at NeurIPS 2020 - yet more on the unreasonable effectiveness of gridded geo-spatial processes. In Proceedings of the NeurIPS 2020 Competition and Demonstration Track (pp. 325–343). PMLR. [[Proceedings]](https://proceedings.mlr.press/v133/kopp21a.html)
 * <a id="T4C2">[2]</a> Kreil, D., Kopp, M., Jonietz, D., Neun, M., Gruca, A., Herruzo, P., Martin, H., Soleymani, A., & Hochreiter, S. (2020). The surprising efficiency of framing geo-spatial time series forecasting as a video prediction task – Insights from the IARAI \t4c Competition at NeurIPS 2019. In Proceedings of the NeurIPS 2019 Competition and Demonstration Track (pp. 232–241). PMLR. [[Proceedings]](https://proceedings.mlr.press/v123/kreil20a.html)
 * <a id="UNet">[3]</a>  Olaf Ronneberger and Philipp Fischer and Thomas Brox (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. CoRR, abs/1505.04597. [[ArXiv]](https://arxiv.org/abs/1505.04597)
 * <a id="fullGNBlock">[4]</a> P. Battaglia, J. Hamrick, V. Bapst, A. Sanchez-Gonzalez, V. Flores Zambaldi, M. Malinowski, A. Tacchetti, D. Raposo, A. Santoro, R. Faulkner, C. Gülccehre, H. Francis Song, A. Ballard, J. Gilmer, G. E. Dahl, A. Vaswani, K. R. Allen, C. Nash, V. Langston, C. Dyer, N. Heess, D. Wierstra, P. Kohli, M. Botvinick, O. Vinyals, Y. Li, R. Pascanu (2018). Relational inductive biases, deep learning, and graph networks. CoRR, abs/1806.01261. [[ArXiv]](https://arxiv.org/abs/1806.01261)
