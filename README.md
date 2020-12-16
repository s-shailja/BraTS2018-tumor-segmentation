# Deepmedic and 3D U-Net for [brain tumor segmentation](https://www.med.upenn.edu/sbia/brats2018/tasks.html)

We created two popular deep learning models DeepMedic and 3D U-Net in PyTorch for the purpose of brain tumor segmentation. The performance of our proposed ensemble on BraTS 2018 dataset is shown in the following table:

|Dataset|Dice(ET)|Dice(WT)|Dice(TC)|Hausdorff95(ET)|Hausdorff95(WT)|Hausdorff95(TC)|
|---|---|---|---|---|---|---|
|Testing|0.782|0.908|0.823|2.96|4.39|6.91|

For more details about our methodology, please refer to our [paper](https://www.frontiersin.org/articles/10.3389/fnins.2019.01449/full)
ET:Enhancing tumor, WT: whole tumor, TC: tumor core.The final trained model for multiclass classification is included along with the results on test dataset.

## Citation

The system was employed for our research presented in [1,2], where the we integrate multiple DeepMedics and 3D U-Nets in order to get a robust tumor segmentation mask. We also utilize the brain parcellation masks for the purpose of bringing the location information to DeepMedic and 3D U-Net. If the use of the software or the idea of the paper positively influences your endeavours, please cite [1,2].

[1] **Po-Yu Kao**, Thuyen Ngo, Angela Zhang, Jefferson Chen, and B. S. Manjunath, "[Brain Tumor Segmentation and Tractographic Feature Extraction from Structural MR Images for Overall Survival Prediction.](https://arxiv.org/abs/1807.07716)"  International MICCAI Brainlesion Workshop. Springer, Cham, 2018.

[2] **Po-Yu Kao**, Shailja Shailja, Jiaxiang Jiang, Angela Zhang, Amil Khan, Jefferson W. Chen, and B. S. Manjunath, "[Improving Patch-Based Convolutional Neural Networks for MRI Brain Tumor Segmentation by Leveraging Location Information](https://www.frontiersin.org/articles/10.3389/fnins.2019.01449/full)" Front. Neurosci. 13:1449. doi: 10.3389/fnins.2019.01449 


## Dependencies

Python3.6
xgboost
