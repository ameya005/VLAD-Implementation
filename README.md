TagMe: Image Categorization
============================
Team: GodFellas
Author: Ameya Joshi - ameya.student@gmail.com 
Requirements: Boost libraries, OpenCV v2.4 or greater

Files and Functions:
--------------------
Training:

1. sorter.cpp - Will sort the images from the training set into folders according to labels. Input required: list of filenames-list.txt
2. surf.cpp - Creates the SURF vectors for each class and aggregates them into xml files per class. Input required: sorted images with list of images in each folder.
3. BOWKtrainer.cpp - Reads the aggregated descriptor xml files and runs k means clustering for getting the vocabulary file. Input: All agggregated Descriptor files in a folder
4. VLAD.cpp- Uses the vocabulary generated along with the descriptors for each image to get the response VLAD vectors for each image. Aggregates them as per class. Input: Vocabulary.xml, list of directories of classes-dir.txt, list of images-list.txt for each directory
5. svm_train.cpp - Trains an RBF SVM and outputs a classifier. Input: Aggregated VLAD vector files in a directory.

Testing:

1. svmTest_VLAD.cpp : Runs the trained classifier and outputs the preicted labels file. Inputs: Classifier file, vocabulary file, validtion images in a folder with a list.txt

Pipeline to use:

sorter->surf->BOWKtrainer->VLAD->svm_train---->svmTest_VLADs

Notes:
1. Please make list.txt files for each folder after sorter.
2. Make a dir.txt file for the set of directories. Example included.
3. Put all necessary xml files generated in one folder with a list.txt to run the code. 

Required file tree

						Root
						 |	
--------------------------------------------------------------------
| | | | |        |            |           |            |           |
1 2 3 4 5 SURFDescriptors SURFVLAD classifier.xml vocabulary.xml   dir.txt
| | | | |        |            |
list.txt(for each directory)

Algorithm is described briefly in the supporting document TagMe.pdf. Please look at the code fiiles for further documentation.


 
