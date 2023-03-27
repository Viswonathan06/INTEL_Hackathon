# INTEL_Hackathon

# **Interview Screening through Personality Detection**
## Introduction and Objective

An automated personality recognition system can be helpful in various applications, such as recruitment, marketing, mental health diagnosis, and personalised content creation. Given it's importance, we develop a **multi-task multi-modal system** for automated personality recognition, trained on **Intel® Optimization for Tensorflow** through the **oneAPI Intel® AI Analytics Toolkit**, that evaluates job applicants using facial landmarks, language, tone, and other communication characteristics. The system provides OCEAN personality values and the likelihood of passing to the interview stage, helping employers evaluate candidates objectively and efficiently. The system assesses the Big five personality traits and determines the probability of qualifying for the interview round, providing a comprehensive behavioral assessment of job applicants.
<br></br>

## oneAPI Modules in-use

1. Intel® Optimization for Tensorflow

Optimized Tensorflow was used to train the Autokeras Multi-task Multi-modal models for OCEAN and interview predictions. 

2. Intel® Optimization for PyTorch

Optimized PyTorch was used for facial landmark extraction through the **3DDFA V2 algorithm**.
<br></br>

## Evaluation

We assume that the oneAPI Intel® AI Analytics Toolkit has been installed in your system. If so, the training and prediction modules will automatically use the Intel® Optimization for Tensorflow and Pytorch.

File included for evaluation are 
1. ```training_prediction.ipynb```
2. ```predict.ipynb```<br>

*These files will run automatically once the Flask application is started. To run them individually, make sure all the requirements mentioned in the **Intel Hackathon Application** have been installed*<br>
To run the application, please navigate to the **Intel Hackathon Application** directory and follow the instructions attached. 

On running the application on the terminal, the following commands should be printed.
![runtime screenshot 1](./Intel%20Hackathon%20Application/terminal_outputs/start.jpeg)<br></br>

Model Imports should have a similar output
![model imports 1](./Intel%20Hackathon%20Application/terminal_outputs/model.jpeg)<br></br>

These outputs reaffirm the usage of oneAPI Intel® AI Analytics Toolkit for training and prediction of the Tensorflow models used. 

## Block Diagram 
![Overall Design](./Intel%20Hackathon%20Application/Design%20Drawings/High%20Level%20Design%20Block%20Diagram.jpeg)


## System Overview 

System 1 : 

* MacBook Pro 14 inch 2021
* Apple M1 Pro Chip - 8-core CPU, 16-core GPU
* 16 GB RAM

System 2 : 

* Lenovo 14 inch
* 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz
* 16 GB RAM





