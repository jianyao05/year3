# year3

# main codes to use are saved_datasets_making, TRAINER_20250102, and INFERENCE_20250102
## image shape for height, width is = 480 x 640
## 20250102
### Task/Action completed:
- created new datasets for both left and right movements for shoulder flexion and cross body arm stretch
- Both contains 50 repetitions of each exercise and was trained with a window size of 60 with step size of 45
### Additional things to note:
- for the creation of datasets, implemented only extracting arm landmarks from the entire body to improve the accuracy and processing of the data within the csv file during prediction\
- Included recording of creating datasets: Able to save the entire video during creating of datasets to be able to review 

## 20250102
### Task/Action completed:
- created datasets for both left and right movements for shoulder flexion, cross body arm stretch, pendulum swing
- Both contains 50 repetitions of each exercise and was trained with a window size of 60 with step size of 45
### Additional things to note:
- for the creation of datasets, implemented only extracting arm landmarks from the entire body to improve the accuracy and processing of the data within the csv file during prediction\
- Included recording of creating datasets: Able to save the entire video during creating of datasets to be able to review

## 20250109
### Task/Action completed:
- <ins>**Current Datasets**</ins>\
[0x]  Armpit Left | [50] Circle Left | [50] Cross Body Left | [100] Pendulum Left | [100] Flexion Left |<br>
[50] Armpit Left | [50] Circle Left | [50] Cross Body Left | [100] Pendulum Left | [100] Flexion Left |
- Updated the codes into a class
- Manage to include the Angle calculations and State sequences for the Counter
- Focused on Left and Right Shoulder Flexion
- Usage of Dot Product to calculate the angles such that the z-axis is also incorporated 
- Try to optimise and include other labels such that it is able to incorporate other exercises so it is able to track by itself
#### Feedback
- Why is the Hand Landmarks (Wrist, Index, Thumbs, etc.) included in the csv file
- Why window step size was chosen to be 45?
- Primarily focus on completing the counter and angle calculation of some exercises, not a must to complete all
- Mentioned that for NYP presentation on FYP and OITP, it is focused more on the Technical sides, hence on our knowledge
and the accuracy and how our figures tha we have chosen came about and be able to explain with reasoning and evidence
on why these numbers are best suited
- Using of Confusion Matrix to test for the accuracy of the model or using preset Libraries 
### Additional things to note:
- for the creation of datasets, implemented only extracting arm landmarks from the entire body to improve the accuracy and processing of the data within the csv file during prediction\
- Included recording of creating datasets: Able to save the entire video during creating of datasets to be able to review

## 20250110
### Task/Action completed
- Feedback on creating a user interface 
- Ability to adjust different angles thresholds and duration of sets or amount of repetitions per set
- Try to justify or remove the hand/finger landmarks within the csv file manually
- Implement a Rest position / Control (user doing random stuff except the exercises)

## 20250121
### Task/Action completed:
- <ins>**Current Datasets**</ins>\
[0x]  Armpit Left | [50] Circle Left | [300] Cross Body Left | [100] Pendulum Left | [300] Flexion Left |<br>
[50] Armpit Right | [50] Circle Right | [300] Cross Body Right | [100] Pendulum Right | [300] Flexion Right |
- Created a User Interface using Streamlit
- Able to adjust the amount of reps and target to reach


## 20250122
### Task/Action completed:
- <ins>**Current Datasets**</ins>\
[0x]  Armpit Left | [300] Circle Left | [300] Cross Body Left | [100] Pendulum Left | [300] Flexion Left |<br>
[50] Armpit Right | [300] Circle Right | [300] Cross Body Right | [100] Pendulum Right | [300] Flexion Right |
- Created a User Interface using Streamlit
- Able to adjust the amount of reps and target to reach
- Created a confusion matrix to test for Model Accuracy
- Created Test Datasets

## 20250123
### Task/Action completed:
- <ins>**Current Datasets**</ins>\
[0x]  Armpit Left | [300] Circle Left | [300] Cross Body Left | [100] Pendulum Left | [300] Flexion Left |<br>
[50] Armpit Right | [300] Circle Right | [300] Cross Body Right | [300] Pendulum Right | [300] Flexion Right |
- Created a User Interface using Streamlit
- Able to adjust the amount of reps and target to reach
- Created a confusion matrix to test for Model Accuracy
- Created Test Datasets
- <ins>**Current Datasets**</ins>\
[0]  Armpit Left | [50] Circle Left | [100] Cross Body Left | [100] Pendulum Left | [100] Flexion Left |<br>
[0] Armpit Right | [50] Circle Right | [100] Cross Body Right | [100] Pendulum Right | [100] Flexion Right |









# Reference Exercises for Frozen Shoulder Rehabilitation
![Frozen shoulder exercise](https://scandinavianphysiotherapycenter.com/wp-content/uploads/2019/09/exercises-to-fix-frozen-shoulder.jpg.webp)
# Reference for MediaPipe Landmarks
![MediaPipe Pose Landmarks](https://miro.medium.com/v2/resize:fit:720/format:webp/1*JJCbfzhTySIqKr1L5pDkpQ.png)