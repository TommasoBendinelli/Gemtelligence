# Gemtelligence: Accelerating Gemstone Classification with Deep Learning
This repository contains the code and resources associated with the research paper "Gemtelligence: Accelerating Gemstone Classification with Deep Learning" authored by Tommaso Bendinelli, Luca Biggio, Daniel Nyfeler, Abhigyan Ghosh, Peter Tollan, Moritz Alexander Kirschmann, and Olga Fink.

### Summary
Gemtelligence is a deep learning-based approach designed to achieve highly accurate and consistent origin determination and treatment detection of gemstones. The system aims to streamline the time-consuming process of human expert visual inspections and reduce dependence on costly analytical methods. It leverages convolutional and attention-based neural networks to process heterogeneous data collected from multiple instruments. This innovative methodology establishes a new benchmark in gemstone analysis by enhancing automation and robustness throughout the entire analytical pipeline.

### Repository Contents
- **src/**: Source code for the Gemtelligence model, including the convolutional and attention-based neural networks.
- **data/**: Sample dataset of gemstone properties and measurements collected using various analytical methods. Available for download: TBD.
- **results/**: Results of the experiments reported in the paper, including the models of each fold. Available for download [https://drive.google.com/file/d/1Zi8Jjr468XgluBgFthjNa3hprZ282MpZ/view?usp=share_link].
- **visualization/**: Code for reproducing the figures presented in the paper and to run the demo
- **workflow/**: Code for training and evaluating the model.
- **plots**: Folder where the figures are saved.

### Getting Started
#### Prerequisites
The code has been tested with Python 3.8.10, CUDA Version 10.2, and Torch 1.8.1 on Ubuntu 20.04.2 LTS using an NVIDIA GeForce RTX 2080 Ti. The code may work with other versions of Python, PyTorch, and CUDA, but compatibility has not been tested.

#### Installation
***Please note that only Python<=3.8 is supported.*** 
1. Create and activate a virtual environment:
```
python3 -m venv env
source env/bin/activate
```
2. Install the appropriate version of PyTorch for your system. To use the same version as the authors, you can install it with:
```
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

3. Install the Gemtelligence package:
```
pip install -e .
```
4. Get the data and the results:
Download the data from the links in the repository content section and then unzip the files in the **data/** and **results/** folders, respectively. 

The **data** folder should contain the following files:
```
cv_splits_heat.json
cv_splits_origin.json
master_file.h5
```
The **results** folder should contain the following files:
```
Heat_Consistency_UV+FTIR
Heat_FTIR
...
Origin_UV+XRF+ICP
Origin_XRF+ICP
```

### Code Structure
The code is organized as follows:
- **src/gemtelligence/learning/hol_net_v1.py**: Implementation of the neural network architecture used in the paper.
- **workflow/training/runner.py**: Entry point for training or evaluating the model.
- **visualization/**: Code for reproducing the figures presented in the paper.

### Gemtelligence Inference Web App
To run the Gemtelligence web app to perform classification of stones, run the following command:
```
streamlit run visualization/demo.py
```

### Gemtelligence Interpretability Web App
To run the Gemtelligence web app to analyze how the model's predictions vary with respect to the input data, run the following command:
```
streamlit run visualization/interpretability.py
```

### Reproduce the Figures in the Paper
To reproduce **Figure 2**, **Figure 3**, and **Figure 4** from the paper, run the following commands. The resulting figures will be saved in the **plots/** folder. The evaluations used for creating the figures are defined in the `experiment_paths` of each file. By default, they are set to the results of the experiments in the `results` folder. To modify them, refer to the next section.

```
python3 visualization/figure_2.py
python3 visualization/figure_3.py
python3 visualization/figure_4.py
```

### Train the Models
To train the model, you need to have a GPU available.

- To train

the model from scratch for origin determination, run the following command:
```
python3 workflow/training/runner.py target=origin sources=["ed","uv","icp"] only_val.uv=True only_val.ed=True only_val.icp=false method.max_epoch=170 use_fix_generator_path_val="data/cv_splits_origin.json"
```
- To train the model from scratch for treatment detection, run the following command:
```
python3 workflow/training/runner.py target=treatment sources=["uv","ftir"] only_val.uv=True only_val.ftir=True method.max_epoch=170 use_fix_generator_path_val="data/cv_splits_heat.json"
```
Both commands will train the model for 170 epochs, assess performance based on the validation set, and evaluate performance on the test set. You can choose which data sources to use for validation and testing by setting the flags `only_val.uv`, `only_val.ed`, and `only_val.icp`. **The results and the model will be saved in the outputs/ folder**.

Note that `ed` refers to EDXRF data, `icp` refers to ICP data, `uv` refers to UV-Vis data, and `ftir` refers to FTIR data.

`use_fix_generator_path_val` specifies which split to use for cross-validation. You can omit the flag `use_fix_generator_path_val` to use a new split.

Your trained models will be located in the **outputs/** folder.

### Evaluate the Models
- To evaluate a model without training it, specify the path with the flag `only_val.path`. For example, the following command evaluates the model saved in the **outputs/2023-05-08/13-06-30 folder**:
```
python3 workflow/training/runner.py target=origin sources=["ed","uv","icp"] only_val.uv=True only_val.ed=False only_val.ftir=false only_val.icp=false only_val.path="'outputs/2023-05-08/13-06-30'" use_fix_generator_path_val="data/cv_splits_origin.json"
```
In the above command, we evaluate the model using only UV data, with the model saved in the **outputs/origin/UV=True_XRF=False_FTIR=False_ICP=False/`date`/`time`/** folder (originally trained with UV, EDXRF, and ICP data). Ensure that `target` and `sources` match the trained model.

Note that each model will be used for each fold in cross-validation. It is important that `use_fix_generator_path_val` matches the one used for training the model; otherwise, the results will be biased, as the model will be evaluated on data seen during training. In the `data` folder, you can find the splits used for origin determination and treatment detection in the files **cv_splits_origin.json** and **cv_splits_heat.json**, respectively.

- To evaluate a model for **Figure 4**, first train a model, then run the evaluation with `is_consistent_exp=True`. For example:
```
python3 workflow/training/runner.py sources=["ed","uv","icp"] stone_type=sapphire target=origin is_consistent_exp=True only_val.path="'results/Origin_UV+XRF/0'" use_fix_generator_path_val="'data/cv_splits_origin.json'"
```
You will get an entry in the `outputs` folder. You can then change the path in `visualization/figure_4.py` to check the results.

### Update the Evaluation for Paper Figures
After obtaining a new evaluation of the model, you can copy the path to the folder into each of the visualization scripts to reproduce the figures in the paper with the newly evaluated model. Update the `experiment_paths` variable accordingly. For example, to reproduce **Figure 2**, replace the values of the dictionary in the lines between 84-87 with your new evaluation:  
Note that the correct combination has to be set using `only_val` during evaluation (see above)

```
experiment_paths = {
    "UV+FTIR": "outputs/2023-05-08/13-06-30/results",  # Replace with the path to your new evaluation
    "XRF": "results/Origin_XRF/0/results",
    "UV": "results/Origin_UV/0/results",
    "UV+XRF": "results/Origin_UV+XRF/0/results"
}
```

After updating the paths, run the corresponding visualization scripts to generate the figures with the new evaluation results.
