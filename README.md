<h1 style="text-align: center;"> Automated Stroke Classification through Convolutional Neural Networks: A Comprehensive Analysis of Blood Clot Images for Improved Treatment Prescription  </h1>
<h3 style="text-align: center;"> IRONHACK - Final project </h3>
<h4 style="text-align: center;"> Sara del Carmen Benítez-Inglott González </h4>

<p align="center">

## State of the Art

An ischemic stroke occurs when a blood clot, known as a thrombus, plugs or blocks an artery which supplies the brain. As a result, the blood flow is restricted, not allowing the gas exchange and nutrition of the organs and tissues [[1](https://my.clevelandclinic.org/health/articles/17060-how-does-the-blood-flow-through-your-heart)]. This happens in critical and risky situations, when the blood blockage reaches the brain. At first, when the tissue is deprived of supply, cells begin to die; however, as time goes on, the likelihood of experiencing a stroke increases, which can lead to permanent brain damages or even death [[2](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10112237/#:~:text=Mortality%20in%20the%20first%2030,on%20early%20treatment(11).)].

In summary, it is a critical **medical emergency** requiring immediate attention.

Numerous factors elevate the likelihood of experiencing an ischemic stroke, including environmental and unmodifiable risk factors [[3](https://www.hopkinsmedicine.org/health/conditions-and-diseases/stroke)]:

Environmental Factors:
- Diabetes
- Smoking
- High blood pressure
- Elevated red blood cell (RBC) count
- Increased levels of cholesterol and lipids in the blood
- Abnormal heart rhythm
- Sedentary lifestyle

Unmodifiable Risk Factors:
- Age-related factors (>65 years old)
- Race
- Gender
- Genetic predisposition
- Transient Ischemic Attacks
- Social and economic factors

## Formation of Blood Clots: Molecular Biology Overview

A blood clot is a partially solid mass that usually forms within blood vessels and is made up of blood cells and other biological constituents. Even though it is the main cause of strokes, it is important to remember that clot formation frequently has a positive function—it stops tissue bleeding when injury occurs [[4](https://my.clevelandclinic.org/health/body/17675-blood-clots)].

Regarding its composition, blood clots are made of: red blood cells (RBCs), fibrin(proteins), platelets (cells from the bone marrow) and white blood cells (WBCs) [[5](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6910231/)].

### CE vs LAA

The two major acute ischemic stokes etilogy subtypes and object of study are: cardiac embolism (CE) and Large Artery Atherosclerosis (LAA). 
They are both provocked by a blood clot, but thereare slightly differencies at the blood clot celulluar level. These differences are:

1. **Cardio Embolism (CE):** RBCs=47.67%, WBCs=4.22%, F= 29.19%, P=18.21%

2. **Artery Atherosclerosis (AA):** RBCs=42.58%, WBCs=3.12%, F=31.31%, P=20.81%


## Aim 

The goal of this project is to classify the blood clot etilogy in ischemic strokes by generating a Convolutional Neural Networks (CNNs).

## Dataset

For this work, a dataset from [Kaggle - Mayo Clinic - STRIP AI](https://www.kaggle.com/competitions/mayo-clinic-strip-ai/code?competitionId=37333&searchQuery=cnn) was used. 
This dataset contain:
- Three folders: 
    - train/ - a folder containing .tif images, used to train the model
    - test/ - a folder containing .tif images used as a test data, to test the model accuracy and error.
    - other/ - a supplemental folder containing .tif images with either unknown etilogy or an etilogy other than CE or LAA. 
- train.csv: containing annotations for images in the train/ folder.

| image-id | center-id | patient-id | image_num | label |
| :------: | :-------: | :--------: | :------:  | :---: |
|  00638_0 |    11     |    00638   |     0     |  CE   |

- test.csv: containing annotations for images in the test/ folder. Has the same columns as the train.csv except for the label classification. 
- other.csv: contains annotations for images in the other/ folder. Has the same columns as test.csv



### Decision-making 
The initial step prior to starting the coding process was to understand the content of the images. Each decision made before and during the pre-processing stage were crucial for achieving a high-accuracy model. The decisions were: 

1. Each image had a greate amount of pixels. For this reason, the images needed to be rescaled. In this case with a normalization of the pixel from 0 (representing black) to 1(representing white), divided by 255 (representing white in the original image).

2. Knowing that each sample was stained with *Martius Scarlet Blue (MSB)* stain, and that each etiology subtype has a slightly different quantity celullar components, it becomes essential for the images pre-processing to not alter the colors. In other words, any mean or standarization of the pixels  in each image is avoided to prevent any unintended changes to the color composition.

3. For the model to learn from each label, the creation of 2 folders, each one with two subfolders had to be done. Apparently, the train_test_split method can be performed in the foloder with images, BUT, it did not clasified them between "CE" or "LAA".  




## Workflow/Pipeline


## Analysis


## Results 


## Conclution

## Interesting literature

The fast development in computer technology and algorithms within the technology sector have allowed the use of these innovations in the field of health, enabling researchers to explore various applications.
Here are some applications in the cardio and neurological field:

[D. Gaddam, C. Mouli and A. Borad, "Early Stage Ischemic Stroke Prediction using Convolution Neural Network," 2022 7th International Conference on Communication and Electronics Systems (ICCES), Coimbatore, India, 2022, pp. 1389-1393, doi: 10.1109/ICCES54183.2022.9835729.](https://ieeexplore.ieee.org/document/9835729)








</p>