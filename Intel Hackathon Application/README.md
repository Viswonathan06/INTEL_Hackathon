# INTEL_Hackathon

# **Interview Screening through Personality Detection**
## Introduction

An automated personality recognition system can be helpful in various applications, such as recruitment, marketing, mental health diagnosis, and personalised content creation. By analysing facial features, language use and communication patterns, the system can identify personality traits and provide valuable insights into human behaviour without any human interaction, thus reducing the bias and probability of human error. 
<br></br>

## Objective

We develop a **multi-task multi-modal system** for automated personality recognition, trained on **Intel® Optimization for Tensorflow** through the **oneAPI Intel® AI Analytics Toolkit**, that evaluates job applicants using facial landmarks, language, tone, and other communication characteristics. The system provides OCEAN personality values and the likelihood of passing to the interview stage, helping employers evaluate candidates objectively and efficiently. The system assesses the Big five personality traits and determines the probability of qualifying for the interview round, providing a comprehensive behavioral assessment of job applicants.
<br></br>

## oneAPI Modules in-use

1. Intel® Optimization for Tensorflow

Optimized Tensorflow was used to train the Autokeras Multi-task Multi-modal models for OCEAN and interview predictions. 

2. Intel® Optimization for PyTorch

Optimized PyTorch was used for facial landmark extraction through the **3DDFA V2 algorithm**.
<br></br>

## Installation

Our code is required a Linux ( preferably Ubuntu distribution ) to function without making any changes. 

    Step 1. Command Line Installation for oneAPI Intel® AI Analytics Toolkit
>```wget https://registrationcenter-download.intel.com/akdlm/irc_nas/19202/l_AIKit_p_2023.1.0.31760_offline.sh```

>```    sudo sh ./l_AIKit_p_2023.1.0.31760_offline.sh```

    Step 2: From the console, locate the downloaded install file.

    Step 3: Use the following command to launch the GUI Installer as the root.
>```$ sudo sh ./<installer>.sh```

    Optionally, use use the following command to launch the GUI Installer as the current user.
>```$ sh ./<installer>.sh  ```

    Step 4: Follow the instructions in the installer.
   
    Step 5: After installation, clone the current directory and install the required modules to run the code smoothly.

>```pip install -r requirements.txt```
    
    Step 6: Run the following command to run the build shell file for the 3DDFA_V2 facial feature extraction algorithm
>```sh build.sh```

    Step 7: The code only needs to be ran on one single file. This can be done in two ways: 
>a. By running the notebook named ```website.ipynb``` **OR**<br></br>
>b. Run the code through the following command <br>
>```python3 website.py```


**The Flask server should start and be accessible at http://localhost:5050/**


    Step 8 :On accessing the webpage, you will be prompted to choose a video of the participant you want to evaluate the personality for. 

    Step 9: Choose the example video already present in the videos directory or move a video that you want to evaluate for. 

**Note: The video has to be present in the videos directory for the application to function properly. Do not include 'video' in the video name.** 

    Step 10: The webpage will now take a few seconds to evaluate and show you the OCEAN and 'interview' values. 
<br></br> 


## Block Diagram 
![Overall Design](./Design%20Drawings/High%20Level%20Design%20Block%20Diagram.jpeg)


## System Overview 

System 1 : 

* MacBook Pro 14 inch 2021
* Apple M1 Pro Chip - 8-core CPU, 16-core GPU
* 16 GB RAM

System 2 : 

* Lenovo 14 inch
* 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz
* 16 GB RAM





