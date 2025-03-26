# FashionImagesClassification_FinalReport
## Identify and Classify Fashion Images and Label them correctly

## Jupyter Notebook with results output (code + output)
https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/FashionProductsClassification_Final_RD.ipynb

## Jupyter Notebook without results output (Just code only)
https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/FashionProductsClassification_Final_RD.ipynb


## DataSet

  ## Label data:
  https://github.com/ravikduvvuri/FashionImagesClassification/blob/main/styles.csv

  ## Images data:
  https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset/data


## 1. Overview and Business Understanding
For a fashion retailer or manufacturer, identifying and entering product attributes into a system is a tedious job; if we have a model that can classify images and get attributes, it is a great help for them and it can also enhance master product data store with less effort. In this project, my goal is explore various models and create a better model to classify images and update label attributes into a dataframe. To Train and Test the models, I leveraged the fashionProductImages Dataset from Kaggle.

## 2. Data Understanding
To understand the data, I did the following:

  ### 3.1 Initial steps done
    
    3.1.1 Imported relevant libraries and packages
    
    3.1.2 Read 'styles.csv' into a pandas dataframe and Loaded product images into /images folder
    
    3.1.3 Displayed sample data using df.sample(5) to see what features are there in the dataset
    
    3.1.4 Gained some domain knowledge and checked data and possible relationships among them
  
  ### 3.2 Next steps done
  
    3.2.1 Using df.info() found the total features, size of the dataset and data types of features
    
    3.2.2 Checked for 'missing data'
    
    3.2.3 Searched for duplicate data
    
  ![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification/blob/main/MissingDataStats.png)

## 4. Data Preparation

### 4.1 Data cleanup
  Based on data analysis, I did the following:

    4.1.1 Data seems clean, so no extensive pre-processing was done

    4.1.2 No duplicates found
    
    4.1.3 Removed missing data rows as they are not needed for images classification
       
### 4.2 EDA on Styles data
  These plots provide the information about how data looks like from distribution perspective
  
    4.2.1 Master Category Data distribution 
    
    4.2.2 Sub Category Data distribution

    4.2.3 Products Data distribution

    4.2.4 Color Distribution by Gender

    4.2.5 Color Distribution by Master Category

    4.2.5 Color Distribution by Usage - Heatmap

![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/MasterCategoryDataDistribution.png)

![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/SubCategoryDataDistribution.png)

![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/ProductTypeDataDistribution.png)

![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/ColorDistributionByGender.png)

![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/ColorDistributionByMasterCategory.png)

![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/HeatMapColorDistributionByUsage.png)

### 4.3 EDA on Images data
  These plots provide the information about how data looks like from distribution perspective
  
    4.3.1 Display sample pictures
    
    4.3.2 Colour distribution of a sample image

![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/DisplaySampleImages.png)

![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/SampleImageColorDistribution.png)

## 5. Engineering Features / Preprocessing of data: Images and Labels

**5.1 Image resize, Label Extraction and data split:**

    5.1.1 Label Extraction to predict classification
    
    5.1.2 Images resize

    5.1.3 Normalize Image data
    
    5.1.3 Split data into Train/Test data

    5.1.4 Display resized sample images

![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/NormalizedSampleImages.png)


## 6. Modeling 

**6.1 Model 1: SVM Model**

    6.1.1 Built SVM model, trained, tested and got F1-Score, Confusion Matrix

    6.1.2 SVC Confusion Matrix

![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/SVC%20Results.png)

**6.2. Model 2: KNN Model**
    
    6.2.1 Built KNN model, trained, tested and got F1-Score, Confusion Matrix

    6.2.1 KNN Confusion Matrix

![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/KNNN%20Results.png)

**6.3. Model 3 : RandomForest Model**
    
    6.3.1 Built RandomForest model, trained, tested and got F1-Score, Confusion Matrix

    6.3.2 RandomForest Confusion Matrix

![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/RFT%20Results.png)

**6.6. Model Comparisions**
    
    6.6.1 Evaluate SVM, KNN, RandomForest results

![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/SVC%20Confustion%20Matrix.png)

![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/KNN%20Confustion%20Matrix.png)

![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/RandomForest%20Confustion%20Matrix.png)

   
## 7.Model Finetuning
    
    7.1.1 Did Hyperparameter tuning
    
    7.1.2 Performed GridSearch
    
    7.1.3 Built Confusion Matrix, Accuracy, Recall, Precision, F1Score

    7.1.4 Shown Best Model and Parameters
   
![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification/blob/main/FineTunedResults.png)


## 8. Additional Packages for Model training 
    
    8.1.1 OpenCV - Model performance evaluation using OpenCV, HOG, LBP via features extraction
    
    8.1.2 Pillow - Model performance evaluation using Pillow and RandomForest

    8.1.3 CNN - Model Performance evaluation

![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/OpenCV%20Results.png)

![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/OpenCV%20Confusion%20Matrix.png)
    
![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/Pillow%20Results.png)

![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/Pillow%20Confusion%20Matrix.png)

![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/CNN%20Results.png)
    
![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/CNN%20Confustion%20Matrix.png)

## 9. Comparision - Actual Label Vs Predicted Label and Plot 
    
    9.1.1 Model performance - Actual Data Vs Predicted Data

![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/PredictedDataComparision.png)

![Alt text](https://github.com/ravikduvvuri/FashionImagesClassification_FinalReport/blob/main/PredictedDataComparisionChart.png)

**Based on models evaluation, Pillow package prediction is perfect with few or no false positives. It classified images correctly. See the confusion matrix above**

## 10. Next Steps and Future Enhancements...
    
## **Now that we created many models and identified two best models (Pillow, CNN) to classify images and label them correctly to ~99% accuracy, the Next Steps would be as follows...**

## 1) Enhance the models to classify and predict labels at a detailed level like 'sub-category' or 'product name'.

## 2) Create an endpoint to leverage it for use in real world application

## 3 Additionally, Enhance the models to extract additional features like color, style, shape, text  during image classification


    

