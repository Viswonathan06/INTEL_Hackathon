
import os
import numpy as np
import pandas as pd
import autokeras as ak
import tensorflow as tf
from tensorflow.keras.models import load_model




def data_selector(suffix, df_train, df_validation, df_test): 
    temp_train = df_train.loc[:, df_train.columns.str.endswith(suffix)]
    temp_validation = df_validation.loc[:, df_validation.columns.str.endswith(suffix)]
    temp_test = df_test.loc[:, df_test.columns.str.endswith(suffix)]
    return temp_train, temp_test, temp_validation


def load_preprocess_data(test_data = None):

    train=pd.read_csv('./Extracted Features/bert_audi_faci_training.csv')
    validation=pd.read_csv('./Extracted Features/bert_audi_faci_validation.csv')
    test=pd.read_csv('./Extracted Features/bert_audi_faci_test.csv')

    info_cols = ['video', 'number']
    dropcols = ['Unnamed: 0.1', 'ethnicity', 'gender', 'Unnamed: 0', 'video_name']
    train_info = train[info_cols]
    validation_info = validation[info_cols]
    test_info = test[info_cols]
    validation_info
    test_info
    train = train.drop(info_cols+dropcols, axis = 1)
    validation = validation.drop(info_cols+dropcols, axis = 1)
    test = test.drop(info_cols+dropcols, axis = 1)

    train = train.append(validation)
    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)

    
    df_train = train.copy(deep = True)
    df_validation = validation.copy(deep=True)
    df_test = test.copy(deep=True)


    # Labels def
    labels = ['OPENMINDEDNESS_Z', 'CONSCIENTIOUSNESS_Z', 'EXTRAVERSION_Z', 'AGREEABLENESS_Z', 'NEGATIVEEMOTIONALITY_Z', 'interview']

    train_O = df_train.pop(labels[0]).to_numpy()
    train_C = df_train.pop(labels[1]).to_numpy()
    train_E = df_train.pop(labels[2]).to_numpy()
    train_A = df_train.pop(labels[3]).to_numpy()
    train_N = df_train.pop(labels[4]).to_numpy()
    train_I = df_train.pop(labels[5]).to_numpy()


    val_O = df_validation.pop(labels[0]).to_numpy()
    val_C = df_validation.pop(labels[1]).to_numpy()
    val_E = df_validation.pop(labels[2]).to_numpy()
    val_A = df_validation.pop(labels[3]).to_numpy()
    val_N = df_validation.pop(labels[4]).to_numpy()
    val_I = df_validation.pop(labels[5]).to_numpy()



    # currently NOT selecting only BERT features
    bt_df_train, bt_df_test, bt_df_validation = data_selector('bt',  df_train, df_validation, df_test)
    audi_df_train, audi_df_test, audi_df_validation = data_selector('audi',  df_train, df_validation, df_test)
    faci_df_train, faci_df_test, faci_df_validation = data_selector('faci',  df_train, df_validation, df_test)

    bt_train_com_np = np.array(bt_df_train)
    bt_val_com_np = np.array(bt_df_validation)
    bt_test_com_np = np.array(bt_df_test)

    audi_train_com_np = np.array(audi_df_train)
    audi_val_com_np = np.array(audi_df_validation)
    audi_test_com_np = np.array(audi_df_test)

    faci_train_com_np = np.array(faci_df_train)
    faci_val_com_np = np.array(faci_df_validation)
    faci_test_com_np = np.array(faci_df_test)

    audio_data = [audi_train_com_np
    ,audi_val_com_np
    ,audi_test_com_np]

    facial_data = [faci_train_com_np
    ,faci_val_com_np
    ,faci_test_com_np]

    bert_data = [
    bt_train_com_np
    ,bt_val_com_np
    ,bt_test_com_np  
    ]


    training = [train_O,
    train_C,
    train_E,
    train_A,
    train_N, 
    train_I]

    validation = [val_O,
    val_C,
    val_E,
    val_A,
    val_N, 
    val_I]

    return audio_data, facial_data, bert_data, training, validation

def training(number_of_trails = 10, epochs=50):
    audio_data, facial_data, bert_data, training, validation = load_preprocess_data()
    OCEAN_models = ['Model_O', 'Model_C', 'Model_E', 'Model_A', 'Model_N']
        
    for i in range(5):
        model = ak.AutoModel(
            inputs=[ak.StructuredDataInput(), ak.StructuredDataInput()],
            outputs=[
                ak.RegressionHead(metrics=["mse"]),
                ak.RegressionHead(metrics=["mse"]),
            ],
            overwrite=True,
            max_trials=5,
        )
        # Fit the model with prepared data.
        model.fit(
            [ audio_data[0], facial_data[0]],
            [training[i], training[5]],
            validation_data=(
                [ audio_data[1], facial_data[1]],
                [validation[i], validation[5]],
            ),
            epochs=50,
        )

        total_model = model.export_model()

        # # Save current model
        total_model.save('./Models/combined_trial_'+OCEAN_models[i])
        # Evaluate on validation set
        # try:
        #     evaluation = model.evaluate([ audio_data[1], facial_data[1]])
        # except:
        #     print("Error in evaluating")
        # # Write loss and error to a file
        # with open('./combined_trial.txt', 'a') as f:
        #     f.write('combined_trial'+' -> ')
        #     f.write(str(evaluation))
        #     f.write('\n')


def prediction():
    audio_data, facial_data, bert_data, training, validation = load_preprocess_data()
    OCEAN_models = ['Model_O', 'Model_C', 'Model_E', 'Model_A', 'Model_N']
    for i in range(5):
        loaded_model = load_model("./Models/combined_trial_"+OCEAN_models[i], custom_objects=ak.CUSTOM_OBJECTS)
        print(loaded_model.summary())
        eval = loaded_model.predict([audio_data[2],facial_data[2]])
        print(eval)
        df1 = pd.DataFrame(eval[0])
        df2 = pd.DataFrame(eval[1])
        df = pd.concat([df1,df2], axis = 1)
        df.to_csv("./predictions/"+OCEAN_models[i]+".csv")
        
def predict_single_video(df_test, vid_name):
    facial_data = df_test.loc[:, df_test.columns.str.endswith('faci')]
    audio_data = df_test.loc[:, df_test.columns.str.endswith('audi')]
    OCEAN_models = ['Model_O', 'Model_C', 'Model_E', 'Model_A', 'Model_N']
    for i in range(5):
        loaded_model = load_model("./Models/combined_trial_"+OCEAN_models[i], custom_objects=ak.CUSTOM_OBJECTS)
        print(loaded_model.summary())
        eval = loaded_model.predict([audio_data,facial_data])
        print(eval)
        df1 = pd.DataFrame(eval[0])
        df2 = pd.DataFrame(eval[1])
        df = pd.concat([df1,df2], axis = 1)
        df.to_csv("./predictions/"+vid_name+"_"+OCEAN_models[i]+".csv")

if __name__ == '__main__':
    prediction()

