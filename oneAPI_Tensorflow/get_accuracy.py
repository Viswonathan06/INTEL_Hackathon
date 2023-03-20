# %%
import pandas as pd
import os
features = {'E':'EXTRAVERSION_Z', 'N':'NEGATIVEEMOTIONALITY_Z','A':'AGREEABLENESS_Z','C':'CONSCIENTIOUSNESS_Z','I':'interview','O':'OPENMINDEDNESS_Z'}

def process_files(path = './oneAPI_Tensorflow/predictions'):
    interview = pd.DataFrame()
    pred = pd.DataFrame()
    for files in os.listdir(path):
        if '.DS_Store' in files:
            continue
        print(files)
        temp = pd.read_csv(os.path.join(path, files))
        # print(temp.columns)
        temp.drop('Unnamed: 0', inplace=True, axis = 1)
        interview = pd.concat([interview, temp.pop('0.1')], axis = 1)
        OCEAN_val = files.split('.')[0].split('_')[1]
        temp.columns = [features[OCEAN_val]]
        pred = pd.concat([pred, temp], axis = 1)
        # print(temp.head())
    return pred, interview
    


    

# %%
def append_interview_value(interview, pred):
    mean= interview.mean( axis=1)
    mean.columns = ['interview']
    # print(mean)
    new_cols =  list(pred.columns)

    pred = pd.concat([pred, mean], axis = 1)
    new_cols.append('interview')
    pred.columns = new_cols
    return pred

# %%
def calculate_accuracy(pred, ground_path = './Extracted Features/bert_audi_faci_test.csv'):
    ground = pd.read_csv(ground_path)
    # print(ground)
    cols = list(features.values())
    ground = ground[cols]
    pred = pred[cols]
    # print(ground)
    # print(pred)
    sub = ground.subtract(pred)
    df = sub.mul(sub)
    df = df.mean(axis = 1)
    print(df.mean())

    


# # %%
# if __name__ == '__main__':
#     pred, interview = process_files()
#     final_pred = append_interview_value(interview, pred)
#     calculate_accuracy(final_pred)



