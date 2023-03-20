# %%
import os
import numpy as np
import pandas as pd
import autokeras as ak
import tensorflow as tf
from tensorflow.keras.models import load_model

# %%
train=pd.read_csv('./Extracted Features/bert_audi_faci_training.csv')
validation=pd.read_csv('./Extracted Features/bert_audi_faci_validation.csv')
test=pd.read_csv('./Extracted Features/bert_audi_faci_test.csv')

# %%
train.index

# %%
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

# %%
test.head()

# %%
train = train.append(validation)
train = train.reset_index(drop=True)
validation = validation.reset_index(drop=True)
test = test.reset_index(drop=True)

# %%
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


# %%
def data_selector(suffix): 
    temp_train = df_train.loc[:, df_train.columns.str.endswith(suffix)]
    temp_validation = df_validation.loc[:, df_validation.columns.str.endswith(suffix)]
    temp_test = df_test.loc[:, df_test.columns.str.endswith(suffix)]
    return temp_train, temp_test, temp_validation

# currently selecting only BERT features
bt_df_train, bt_df_test, bt_df_validation = data_selector('bt')
audi_df_train, audi_df_test, audi_df_validation = data_selector('audi')
faci_df_train, faci_df_test, faci_df_validation = data_selector('faci')

# %%//
bt_train_com_np = np.array(bt_df_train)
bt_val_com_np = np.array(bt_df_validation)
bt_test_com_np = np.array(bt_df_test)

audi_train_com_np = np.array(audi_df_train)
audi_val_com_np = np.array(audi_df_validation)
audi_test_com_np = np.array(audi_df_test)

faci_train_com_np = np.array(faci_df_train)
faci_val_com_np = np.array(faci_df_validation)
faci_test_com_np = np.array(faci_df_test)

OCEAN_models = ['Model_O', 'Model_C', 'Model_E', 'Model_A', 'Model_N']

training = [train_O,
train_C,
train_E,
train_A,
train_N]

validation = [val_O,
val_C,
val_E,
val_A,
val_N]

# %%
# train_com_np.shape


    # %%
for i in range(5):
    # model = ak.AutoModel(
    #     inputs=[ak.StructuredDataInput(), ak.StructuredDataInput(), ak.StructuredDataInput()],
    #     outputs=[
    #         ak.RegressionHead(metrics=["mse"]),
    #         ak.RegressionHead(metrics=["mse"]),
    #     ],
    #     overwrite=True,
    #     max_trials=5,
    # )
    # # Fit the model with prepared data.
    # model.fit(
    #     [ audi_train_com_np, faci_train_com_np],
    #     [training[i], train_I],
    #     validation_data=(
    #         [ audi_val_com_np, faci_val_com_np],
    #         [validation[i], val_I],
    #     ),
    #     epochs=50,
    # )

    # total_model = model.export_model()

    # # Save current model
    # total_model.save('./combined_trial'+OCEAN_models[i])
    # # Evaluate on validation set
    # try:
    #     evaluation = model.evaluate([ audi_val_com_np, faci_val_com_np])
    # except:
    #     print("Error in evaluating")
    # Write loss and error to a file
    # with open('./combined_trial.txt', 'a') as f:
    #     f.write('combined_trial'+' -> ')
    #     f.write(str(evaluation))
    #     f.write('\n')


    loaded_model = load_model("./Models/combined_trial"+OCEAN_models[i], custom_objects=ak.CUSTOM_OBJECTS)
    print(loaded_model.summary())
    eval = loaded_model.predict([bt_test_com_np, audi_test_com_np,faci_test_com_np])
    print(eval)
    df1 = pd.DataFrame(eval[0])
    df2 = pd.DataFrame(eval[1])
    df = pd.concat([df1,df2], axis = 1)
    df.to_csv("./predictions/"+OCEAN_models[i]+".csv")
# # %%
# """# Train
# Currently:  
# Validation split: 0.15  
# Epochs: 1000  
# Trials: 100  
# ## Combined models
# """
# print("Combined TRAINING BEGINS")
# model_names = ['Model_O','Model_C','Model_E','Model_A','Model_N']

# for i in range(5):
#   if i < 4 : 
#     continue
#   train = comb_train_set[i]
#   val = comb_validation_set[i]
#   print(model_names[i])
#   # Define a regressor
#   total_reg = ak.StructuredDataRegressor(max_trials=100, overwrite=True,project_name = 'combined'+model_names[i], directory = './Dump Data')
#   # Feed the tensorflow Dataset to the regressor.
#   total_reg.fit(train, epochs=1000, validation_split=0.15)
#   # Convert to model   
#   total_model = total_reg.export_model()
#   # Evaluate on validation set
#   evaluation = total_reg.evaluate(val)
#   # Write loss and error to a file
#   with open('./Dump Data/combined_eval_val_debiased_gender.txt', 'a') as f:
#       f.write('combined'+model_names[i]+' -> ')
#       f.write(str(evaluation))
#       f.write('\n')
#   # Save current model
#   total_model.save('./Debiased_Models/FaceBodyModels/Combined/'+model_names[i])


# # %% [markdown]
# # # Prediction
# # 

# # %%

# # Change this according to the path of male and female
# old_path = './Models/FaceBodyModels/Combined/'
# young_path = './Models/FaceBodyModels/Combined/'

# # change this to whatever you have named the folders where each model is inside male and female
# OCEAN_models = ['Model_O', 'Model_C', 'Model_E', 'Model_A', 'Model_N']
# # Make sure they have similar names
# for mod in OCEAN_models:
#     model = tf.keras.models.load_model(old_path+mod)
#     model.summary
#     prediction = model.predict(M_test_np)
#     print(len(prediction))
#     # print(len(prediction[0]))
#     prediction = np.array(prediction)
#     print(prediction.shape)
#     M_test_data = pd.concat([M_test_data, pd.DataFrame(prediction)], axis=1)
# # Here F_test_data is only the ID and minute columns of the test data you're passing in, which we are merging the 
# # predictions with. 
# # Make sure you change it to whatever your dataframe is.

# # F_test_data.to_csv("./Results/fb_pred.csv")

# """# Making predictions based on female models for OCEAN individually"""

# for mod in OCEAN_models:
#     model = tf.keras.models.load_model(young_path+mod)
#     prediction = model.predict(F_test_np)
#     print(len(prediction))
#     prediction = np.array(prediction)
#     F_test_data = pd.concat([F_test_data, pd.DataFrame(prediction)], axis=1)

# total_test_data = F_test_data.append(M_test_data)
# print(total_test_data.columns)
# total_test_data=total_test_data.sort_values(by=['ID_y'])
# total_test_data.to_csv("./Results/fb_pred_gend.csv")
# print("Prediction over")

# # %%
# total_test_data.head()



# # %%
# df_fb = total_test_data.reset_index(drop= True)

# df_fb = df_fb.drop(["minute"], axis = 1)
# # mean_df_fb = df_fb.groupby(['ID_y'], as_index=False).mean()
# df_fb.columns = [0, 1, 2, 3, 4, 5]

# # %%
# df_fb.head()

# # %%
# df_fb.to_csv("./Results/minute_wise/fb_minwise.csv")

# # %%



