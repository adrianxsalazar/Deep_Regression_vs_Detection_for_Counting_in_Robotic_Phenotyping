# Deep Regression versus Detection for Counting in Robotic Phenotyping Code.

This repository contains the implementation of the paper
"Deep Regression versus Detection for Counting in Robotic Phenotyping."
This implementation contains the code to run the two kinds of approaches we use in
our paper; counting by regression and counting by detection.

Both approaches follow the same directory structure. However, the libraries and
commands required to run them differ. This page explains how to run both
methodologies, but for more detailed description check the following links:

<ul>
 <li>Counting by detection detailed page: <a href="https://github.com/adrianxsalazar/density_based_methods_counting">Counting by detection.</a></li>
 <li>Counting by regression detailed page:<a href="https://github.com/adrianxsalazar/faster-r-cnn-implementation"> Counting by regression.</a>.</li>
</ul>

What can you do with this implementation?
<ul>
 <li> Train object detection models with your custom datasets or with the datasets proposed in our paper.</li>
 <li> Train counting by regression approaches with your custom datasets or with the datasets proposed in our paper.</li>
 <li> Training and testing with few command.</li>
</ul>


</h4> An example of the output of the detection approaches. </h4>
<p class="aligncenter">
<img src="https://github.com/adrianxsalazar/faster-r-cnn-implementation/blob/master/readme_images/detection_sample.png" alt="detection sample">
</p>

<h4> An example of the output of the density-based counting approaches. </h4>
<p class="aligncenter">
<img src="https://github.com/adrianxsalazar/density_based_methods_counting/blob/master/readme_images/output.png" alt="detection sample">
</p>


<h3> Directory structure </h3>

First, you need to follow the given directory structure.
You can always change the code if you do not want to follow this structure.
However the commands we use will not work. I will explain later which lines of
code you need to change in case you want.


```

project                                           #Project folder. Typically we run our code from this folder.
│   README.md                                     #Readme of the project.
│
└───code                                          #Folder where we store the code.
│   │
|   └───models
|   │   └───can                                   #Folder that contains the code to run can.
|   │   |   │   train.py                          #File with the code to train a can model.
|   │   |   │   test.py                           #File with the code to test a can model.
|   |   |   |   dataset.py                        #File with the code with the data loader. We do not use this file directly.
|   |   |   |   image.py                          #This code contains is in charge of modifying the images we use. We do not use this file directly.
|   |   |   |   utils.py                          #Several tools we use in the training process. We do not use this file directly.
|   |   |   |   model.py                          #The code contains the model. We do not use this file directly.
|   |   |   
|   |   └───CSRNet                                #Folder that contains the code to run CSRNet.
|   │       │   train.py                          #File with the code to train a CSRNet model.
|   │       │   test.py                           #File with the code to test a CSRNet model.
|   |       |   dataset.py                        #File with the code with the data loader. We do not use this file directly.
|   |       |   image.py                          #This code contains is in charge of modifying the images we use. We do not use this file directly.
|   |       |   utils.py                          #Several tools we use in the training process. We do not use this file directly.
|   |       |   model.py                          #The code contains the model. We do not use this file directly.
|   |     
│   └───faster_rcnn                               #Folder that contains the code to train and test the faster R CNN
│   |       │   faster_rcnn.py                    #faster rcnn training code.
│   |       │   testing_faster_rcnn.py            #faster rcnn testing code.
|   |       └───tools
|   |            |   decision_anchors.py          #K-means approach to choose anchor sizes.
|   |            |   plot_results_faster_rcnn.py  #Plot the results of the trained models.
│   |
|   └───utils                                     #This folder contains tools to train density-based models.
|       |   creation_density_maps.py              #Code to create the ground truth density maps.
|       |   json_files_creator.py                 #Code to create the .json file with the path of the images we want to use for training, testing, and validation.
|
└───datasets                                      #Folder where we save the datasets.
|   │   ...
|   └───dataset_A                                 #Dataset folder. Each dataset has to have a folder.
|       |   density_test_list.json                #.json file that contains a list of the paths of the testing images.
|       |   density_train_list.json               #.json file that contains a list of the paths of the training images.
|       |   density_val_list.json                 #.json file that contains a list of the paths of the validation images.
|       |   json_test_set.json                    #COCO JSON annotation file of the testing images.
|       |   json_train_set.json                   #COCO JSON annotation file of the training images.
|       |   json_val_set.json                     #COCO JSON annotation file of the validation images.
|       |
|       └───all                                   #Folder where we place the images and the ground truth density maps.
|           | img_1.png                           #Image we are using
|           | img_1.h5                            #ground truth density map
|           | img_2.png                           #""
|           | img_2.h5                            #""
|           | ...
|   
└───saved_models                                  #Folder where we save the models.
    |   ...
    └───can                                       #Folder where we save the models as a result the can training process.
    |   |   ...
    |   └───dataset_A                             #Folder where we save the models we trained using dataset A.
    |       └───best_model.pth                    #Model we get from training for can.
    |
    └───CSRNet
    |   |   ...
    |   └───dataset_A                             #Folder where we save the models we trained using CSRNet on dataset A.
    |       └───best_model.pth                    #Model we get from training for CSRNet.
    |
    └───faster_cnn                                
        |   ...
        └───dataset_A                             #Folder where we save the models we trained using faster RCNN on dataset A.
            └───best_model.pth                    #Model we get from training with faster RCNN.


```

<h3> Running the code for faster RCNN </h3>

Before explaining how to use this implementation, I should point to the detectron2
framework. Detectron2 is a fantastic tool for object detection and segmentation.
You can get more information about this framework in the official
<a href="https://github.com/facebookresearch/detectron2">repository.</a>.
I recommend to check their website to install the library in your device.
If you want to know more about faster R-CNN, I recommend to start with the
original article:
"Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks".

Once the directory is ready, place the training, testing, and validation coco JSON
files in the datasets/<name_of_your_dataset>/ directory.
Then you only have to rename them as "json_train_set.json", "json_test_set.json",
and "json_val_set.json". Then, copy all the dataset images under the directory
datasets/<name_of_your_dataset>/all/. Now, everything is ready to train our
Faster R-CNN.


To train the model,  we need to run the following command in our terminal in
the project folder.

```

$ python3 code/models/faster_rcnn/faster_rcnn.py

```

This command will not work. We need to indicate which dataset we want to use and
the folder's name to store the trained models. We can register these elements
with the commands "-dataset" and "-model_output". If our dataset name is "dataset A"
and the folder's name where we want to store the model is "dataset A output",
the new command will be as follows.

```

$ python3 code/models/faster_rcnn/faster_rcnn.py -dataset "dataset A" -model_output "dataset A output"

```

The following command trains a Faster RCNN in the "dataset A". The learning rate
is 0.0002 and a patience of 20. A patience of 20 means that if the model does not
improve in 20 validations checks, the training will stop.

```

$ python3 code/models/faster_rcnn/faster_rcnn.py -dataset "dataset A" -model_output "dataset A output" -learning_rate 0.0002 -patience 20


```

<h3> Running the code for the density-based models. </h3>
Before training your density based models, make sure that you have installed
PyTorch in your device.

For training, all the implemented density-based methods require of three
parameters; a first parameter indicating the path of a ".json" that indicates
the location of the ground truth training images, the second parameter is another
".json" that does the same as the previous one but indicates the location of the
validation items, and the last element is a path indicating the folder where the
trained models are saved. The following example trains a CSRNet model on
a dataset called "dataset_A"

```

$ python code/models/CSRNet/train.py "./datasets/dataset_A/density_train_list.json"  "./datasets/dataset_A/density_val_list.json" "./saved_models/CSRNet/dataset_A/"


```

The following example trains a "CAN" model

```

$ python code/models/can/train.py  "./datasets/dataset_A/density_train_list.json"  "./datasets/dataset_A/density_val_list.json" "./saved_models/can/global-wheat-detection/"


```

<h3> Testing the Faster R-CNN model. </h3>
Once we finish with the Faster R-CNN training, we can evaluate our model.
The python file "testing_faster_rcnn.py" contains the code to test our models.
We only need to run this file with the corresponding commands "-model" and "-model_output".
The testing uses the model located in the "model_output".
We will need to use the "model_output" we indicated during the training process.
The following command tests the detection model that we trained before.

```

$ python3 code/models/faster_rcnn/testing_faster_rcnn.py -dataset "dataset A" -model_output "dataset A output"


```


<h3> Testing the density based model. </h3>
As with the testing of the detection model, we can easily evaluate our density-based
counting model. The following command tests the can model that we trained before.

```

$ python3 code/models/can/test.py -test_json "./datasets/dataset_A/density_test_list.json" -output "./saved_models/can/dataset_A/"


```



<h3> Training parameters for Faster RCNN </h3>

```

-> -model: description=standard model used for training,
            required=False, default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", type=str

-> -check_model: description=check if there is a checkpoint from previous trainings processes,
            required=False, default=False, type=bool

-> -model_output: description=where the model is going be stored,
            required=False, default="dataset_A", type=str

-> -dataset: description=dataset to use,
            required=False, default="dataset_A", type=str

-> -standard_anchors: description=True if we want to use the standard anchor sizes, False if we want to suggest ours,
            required=False, default=True, type=bool

-> -learning_rate: description=learning rate in the training process
            required=False, default=0.0025, type=float

-> -images_per_batch: description=number of images used in each batch,
            required=False, default=6, type=int

-> -anchor_size: description= if -standard_anchors is True, the size of the anchors in the rpn,
            required=False, default='32,64,128,256,512', type=str

-> -aspect_ratios: description= if -standard_anchors is True, this indicates the aspect ration to use in the rpn
            required=False, default='0.5,1.0,2.0', type=str )

-> -roi_thresh: description=Overlap required between a ROI and ground-truth box in order for that ROI to be used as training example,
            required=False, default=0.5, type=float

-> -number_classes: description=number of classes,
            required=False, default=1, type=int

-> -evaluation_period: description= The command indicates the number of epochs required to evaluate our model in the validations set,
            required=False, default=5, type=int)

-> -patience: description= Number of evaluations without improvement required to stop the training process,
            required=False, default=20, type=int

-> -warm_up_patience: description=Number of evaluations that will happen independently of whether the validation loss improves,
            required=False, default=20, type=int


```


<h3> Choose the anchor size with k-means </h3>
We set up a command to choose the anchor size in the region proposal network. However, it might difficult to select the right size to improve your detection models. An approach to get the right anchor size is to cluster the bounding boxes of our dataset. Then we can find representative groups of bounding box sizes. We can use the representative centroids of the clustering process as the anchors in the rpm. The Yolo V2 paper proposed this method that led to improvements in the Yolo performance. Now, we can implement this in our faster R-CNNs.

To obtain the centroids, we need to run the following command. The parameter "-json_name" needs the name of the COCO JSON with the dataset labels. The "-dataset_path" command requires the path of the COCO JSON.  We can indicate the number of clusters with the attribute "-number_clusters".

```

$ python3 code/faster_rcnn/tools/decision_anchors.py -json_name "json_train_set.json" -dataset_path "./datasets/dataset A/" -number_cluster 9

```
