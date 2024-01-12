# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC Generic classifier with Persistance on Spark & Probabilities

# COMMAND ----------

# DBTITLE 1,Argument Description
# MAGIC %md
# MAGIC
# MAGIC * **historic_table_name** - _[String]_ he table name having historic(training) data.
# MAGIC * **prediction_table_name** - _[String]_ The table name having data for prediction.
# MAGIC
# MAGIC * **input_from** - _[String]_ Database source of training and predicting data. Can be in (Spark,Redshift)
# MAGIC * **output_to** - _[String]_ Database source for storing the results. Can be in (Spark,Redshift)
# MAGIC
# MAGIC * **model_id** - _[String]_ The identifier of the model being currently used in generic clasifier.
# MAGIC * **id_name** - _[String]_ The id variable column name.
# MAGIC * **target_name** - _[String]_ The taget variable column name.
# MAGIC * **accuracy_focus** - _[String]_ The metric on which the best model is chosen. Can be in (accuracy,precision,recall)
# MAGIC * **target_focus** - _[String]_ The target category for which accuracy metrics are calculated for evaluation.
# MAGIC
# MAGIC  
# MAGIC * **is_gridsearch** - _[Boolean]_ set True to trigger gridsearch in hyperparameter tuning. Set False to trigger hyperparameter tuning with default parameters.
# MAGIC * **is_update_model_param** - _[Boolean]_ set True to update the hyperparameters stored for a model_id. set False to skip updation.
# MAGIC * **persist_model** - _[Boolean]_ set True to persist the trained model as a pickle file. set False to skip persistence.
# MAGIC * **read_model** - _[Boolean]_ set True to predict using an already persisted model. set False to train the model and then do the predictions.
# MAGIC
# MAGIC
# MAGIC * **preferred_model** - _[String]_ set it as (preferred_model_type + "_" + preferred_sampling_type) to fix the model_type and sampling_type for the model training. set it as _empty_ to run for all model_types and sampling_types combinations and then choose the best.
# MAGIC  
# MAGIC * **persist_cl_report** - _[String]_ set it to the table name to persist the classification report of training data. Set as False to not persist.
# MAGIC * **persist_cv_score** - _[String]_ set it to the table name to persist the cross validation score report of training data. Set as False to not persist.
# MAGIC * **persist_precision_buckets** - _[String]_ set it to the table name to persist the bucketwise precision scores of training data. Set as False to not persist.
# MAGIC

# COMMAND ----------

# MAGIC %run
# MAGIC /Users/vatsals@playsimple.in/generic_pre_run

# COMMAND ----------

import warnings 
warnings.filterwarnings("ignore")

# COMMAND ----------

# DBTITLE 1,Libraries used
import pandas as pd
import numpy as np
import json
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE


from IPython.display import  HTML
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score, accuracy_score, plot_confusion_matrix, multilabel_confusion_matrix
import random
from sklearn.model_selection import cross_val_score,cross_validate

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
from sklearn.preprocessing import OrdinalEncoder

from python_sdk.utils.logging import initLoggingWithLogPath
import logging

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

import pickle as pk
import datetime as dt

import mlflow
from pathlib import Path

import datetime

# COMMAND ----------

print (imblearn.__version__)
print(pd.__version__)

# COMMAND ----------

# DBTITLE 1,Read Run Time Arguments
try:
  historic_table_name = dbutils.widgets.get("historic_table_name")
  prediction_table_name = dbutils.widgets.get("prediction_table_name")
  target_name = dbutils.widgets.get("target_name")
  id_name = dbutils.widgets.get("id_name")
  accuracy_focus = dbutils.widgets.get("accuracy_focus")
  target_focus = dbutils.widgets.get("target_focus").split(",")
  log_path = dbutils.widgets.get("log_path")
  input_from = dbutils.widgets.get("input_from")
  output_to = dbutils.widgets.get("output_to")
  preferred_model = dbutils.widgets.get("preferred_model")
  show_probability = eval(dbutils.widgets.get("show_probability"))
  cv_score_persistence_table = str(dbutils.widgets.get("persist_cv_score")).lower()
  cl_report_persistence_table = str(dbutils.widgets.get("persist_cl_report")).lower()
  prec_buckets_persistence_table = str(dbutils.widgets.get("persist_precision_buckets")).lower()
  
  
  if((preferred_model == "None") | (preferred_model == "-" ) | (preferred_model == "")):
    preferred_model = ""
    preferred_sampling = ""
  else:
    preferred_sampling = preferred_model.split("_")[1]
    preferred_model = preferred_model.split("_")[0]
  
  is_gridsearch = eval(dbutils.widgets.get("is_gridsearch"))
  print("is_gridsearch. ",is_gridsearch)
    
  is_update_model_param=eval(dbutils.widgets.get("is_update_model_param"))
  print("is_update_model_param  ",is_update_model_param)
  
  model_ID=dbutils.widgets.get("model_ID")
  print("model_ID  ",model_ID)
  
  persist_model = eval(dbutils.widgets.get("persist_model"))
  print("persist_model  ",persist_model)

  read_model = eval(dbutils.widgets.get("read_model"))
  print("read_model  ",read_model)

  path = dbutils.widgets.get("model_location")
  if path == "":
    path = "/dbfs/datascience/"
  print("model_location  ",path)

  try:
    cv_kfold = int(dbutils.widgets.get("cv_kfold"))
  except:
    cv_kfold=5

  print("cv_kfold. ",cv_kfold)

  print(log_path)
  primary_keys = [id_name]

  if(log_path):
    initLoggingWithLogPath(log_path=log_path)
    
  logging.getLogger("py4j").setLevel(logging.ERROR)
  logging.info("Job started")
except Exception as e:
  logging.error(e)
  logging.shutdown()
  raise Exception("Run time args")

# COMMAND ----------

# DBTITLE 1,DS Constants
class_imbalance_percentage = 0.10
under_sampling_criteria_count = 10000
# model_names = ['rf', 'dt', 'xg', 'nb', 'ovr']
model_names = ['rf','xg']

if(preferred_model != ""): #overwrite the models to fix it to the preferred model
  model_names = [preferred_model]


# COMMAND ----------

# def start_mlflow_new_experiment(exp_name,run_name):
#   try:
#     mlflow.end_run()
#     exp_id = mlflow.set_experiment(exp_name).experiment_id
    
#     mlflow.start_run(experiment_id=exp_id, run_name = run_name)

#     mlflow.log_params({
#     "historic_table_name":historic_table_name,
#     "prediction_table_name":prediction_table_name,
#     "target_name":target_name,
#     "id_name":id_name,
#     "target_focus":target_focus,
#     "log_path":log_path,
#     "input_from":input_from,
#     "output_to":output_to,
#     "run_ID":run_ID
#     })
#     # logging.info("MlFlow started")
#   except Exception as e:
#     # logging.error(e)
#     # logging.shutdown()
#     raise Exception("Error in starting ML Flow")

# run_name = 'general_params'
# exp_name="/Shared/generic_classifier_"+ str(model_ID) + '_' + str(datetime.datetime.today())
# print("exp_name. ",exp_name)
# start_mlflow_new_experiment(exp_name=exp_name,run_name=run_name)

# COMMAND ----------

try:
  mlflow.end_run()
  exp_id = mlflow.set_experiment("/Shared/generic_classifier_"+ str(model_ID) + '_' + str(datetime.datetime.today())).experiment_id
  run_name = "general_params"
  mlflow.start_run(experiment_id=exp_id, run_name = run_name)

  mlflow.log_params({"accuracy_focus":accuracy_focus,"historic_table_name":historic_table_name,"id_name":id_name,"input_from":input_from,"is_gridsearch":is_gridsearch,"is_update_model_param":is_update_model_param,"log_path":log_path,"model_id":model_ID,"model_location":path,"output_to":output_to,"persist_cl_report":cl_report_persistence_table,"persist_cv_score":cv_score_persistence_table,"persist_model":persist_model,"persist_percision_buckets":prec_buckets_persistence_table,"prediction_table_name":prediction_table_name,"preferred_model":preferred_model,"read_model":read_model,"show_prob":show_probability,"target_name":target_name,"target_focus":target_focus})
  logging.info("MlFlow started")
except Exception as e:
  logging.error(e)
  logging.shutdown()

# COMMAND ----------

# DBTITLE 1,Classification Steps
try:
  displayHTML("A. Data Exploration & Pre processing" 
              + "<br><br> B. Feature Engineering"
              + "<br><br> C. Model Building  "
              + "<br><br> D. Model Evaluation "
              + "<br><br> E. Model Prediction & Persist results"
              )
except Exception as e:
  logging.error(e)
  logging.shutdown()
  raise Exception("Init")
  

# COMMAND ----------

# MAGIC %md
# MAGIC ## read data

# COMMAND ----------

# DBTITLE 1,1. b. Read Prediction from RS/Spark Table
if(input_from == "redshift"):
  try:
    from notebooks.external_services.db.redshift import NotebooksRedshiftUtil
    rs_obj = NotebooksRedshiftUtil()
    pred_df_orig = rs_obj.fetch_all_as_df("select * from " + prediction_table_name)
    if(target_name in pred_df_orig.columns):
      pred_df_orig[target_name+'_actual'] = pred_df_orig[target_name]
    else:
      pred_df_orig[target_name+'_actual'] = 'NA'
    
    original_colnames = pred_df_orig.columns
    print("Input from redshift")
  except Exception as e:
    logging.error(e)
    logging.shutdown()
    raise Exception("Read error")
    
elif(input_from == "spark"):
  try:
    spark_pred_df =spark.sql("select * from "+ prediction_table_name)
    if(target_name in spark_pred_df.columns):
      spark_pred_df=spark_pred_df.withColumn(target_name+'_actual',spark_pred_df[target_name])
    else:
      spark_pred_df=spark_pred_df.withColumn(target_name+'_actual',F.lit('NA'))
      
    pred_df_orig = spark_pred_df.toPandas()
    display(pred_df_orig)
    original_colnames = pred_df_orig.columns
    logging.info("Read Prediction data")
    logging.info(spark_pred_df.printSchema)
  except Exception as e:
    logging.error(e)
    logging.shutdown()
    raise Exception("Read error")

# COMMAND ----------

# DBTITLE 1,1.a.Read Historic data from RS/Spark Table
if(input_from == "redshift"):
  try:
    rs_obj = NotebooksRedshiftUtil()
    input_df_orig = rs_obj.fetch_all_as_df("select * from " + historic_table_name)
    logging.info("Read training data")
  except:
    input_df_orig = pd.DataFrame(data=None, columns=pred_df_orig.columns) #copy schema of prediction data into historic data
    ## needed in case read_model=True and only pred data is passed 
    print("Historic table empty")
    
elif(input_from == "spark"):
  try:
    spark_hist_df=spark.sql("select * from "+ historic_table_name)
    input_df_orig = spark_hist_df.toPandas()
    display(input_df_orig.head())
    logging.info("Read training data")
  except:
    input_df_orig = pd.DataFrame(data=None, columns=pred_df_orig.columns) #copy schema of prediction data into historic data
    ## needed in case read_model=True and only pred data is passed 
    print("Historic table empty")

# COMMAND ----------

print(input_df_orig.shape)
print(pred_df_orig.shape)
# one extra column in prediction df being the target_actual

# COMMAND ----------

input_df_orig=input_df_orig.sort_values(by=[id_name])

# COMMAND ----------

input_df_orig.head()

# COMMAND ----------

# DBTITLE 1,1. c. Target & ID Preparation
try:
  input_df = input_df_orig.copy() # Making a copy so that it need not be read everytime code is changed
  pred_df = pred_df_orig.copy()
  
  # converting target column type to str to make it work across different scenarios , also target focus given will be taken as string
  input_df['target'] = input_df[target_name].astype(str)
  pred_df['target'] = pred_df[target_name].astype(str)
  if (target_name != "target"):
    input_df = input_df.drop([target_name], axis = 1)
    pred_df = pred_df.drop([target_name], axis = 1)
  
  input_df['id'] = input_df[id_name]
  pred_df['id'] = pred_df[id_name]
  if (id_name != "id"):
    input_df = input_df.drop([id_name], axis = 1)
    pred_df = pred_df.drop([id_name], axis = 1)

  #Re-Ordering Columns
  cols = (['id']) + (input_df.drop(['id', 'target'], axis = 1).columns.tolist())  + ['target']
  input_df = input_df[cols]
  pred_df = pred_df[cols]
  display(pred_df.head())
except Exception as e:
  logging.error(e)
  logging.shutdown()
  raise Exception("Target prep error")

# COMMAND ----------

# DBTITLE 1,Target Value Counts
print("train data target value counts - ")
print(input_df['target'].value_counts(normalize=True))
print("----------------------------------")
print("test data target value counts - ")
print(pred_df['target'].value_counts(normalize=True))

# COMMAND ----------

# DBTITLE 1,9.b  Number of unique categories in target.
try:
  target_classes = input_df.target.unique().size
  print("target_classes :: ",target_classes)
except Exception as e:
  logging.error(e)
  logging.shutdown()

# COMMAND ----------

try:
  if(target_focus[0] == 'all'):
    target_lst = input_df['target'].unique()
  else:
    target_lst = [i for i in target_focus]
    print("target_focus  ::  ",target_focus)
    print("target_lst  ::  ",target_lst)
except Exception as e:
  logging.error(e)
  logging.shutdown()
  raise Exception("FE error")
  

# COMMAND ----------

# DBTITLE 1,10. Class Imbalance

try:
  print("class_imbalance_percentage  ::  ",class_imbalance_percentage)
  class_imbalance = False
  class_imbalance_extent = round((input_df['target'].value_counts().min()/input_df['target'].value_counts().max()), 4)

  if(class_imbalance_extent < class_imbalance_percentage):
    class_imbalance = True

  displayHTML("Class Imbalance Proportion between majority and minority classes : " + str(class_imbalance_extent) + ":1" +"<br><br> Class Imbalance : " + str(class_imbalance)  )

  mlflow.log_param("class_imbalance",class_imbalance)
  
except Exception as e:
  logging.error(e)
  logging.shutdown()
  raise Exception("CI error")
  

# COMMAND ----------



print("train data target value counts - ")
print(input_df.shape)
print(input_df['target'].value_counts(dropna=False,normalize=True)*100)

print("----------------------------------")
print("test data target value counts - ")
print(pred_df.shape)
print(pred_df['target'].value_counts(dropna=False,normalize=True)*100)

# COMMAND ----------

# DBTITLE 1,Generic Functions for model prediction
## KEEP IN GC
def predict_target(model_fit, x_val):

  X_val_df=x_val.values
  if('id' in x_val.columns):
    X_val_df = x_val.drop(['id'], axis = 1).values
  if(id_name in x_val.columns):
    X_val_df = X_val_df.drop([id_name], axis = 1).values
    
  y_pred = model_fit.predict(X_val_df)
  del X_val_df
  return y_pred
  
def predict_target_prob(model_fit, x_val, class_names):

  X_val_df=x_val.values
  if('id' in x_val.columns):
    X_val_df = x_val.drop(['id'], axis = 1).values
  if(id_name in x_val.columns):
    X_val_df = X_val_df.drop([id_name], axis = 1).values
    
  y_pred = pd.DataFrame(model_fit.predict_proba(X_val_df),columns = class_names)
  del X_val_df
  return y_pred

# COMMAND ----------

# DBTITLE 1,Feature and Target Encoding Function

#prepare target
def target_encoding(y_train, y_test):
  le = LabelEncoder()
  le.fit(y_train)
  y_train_enc = le.transform(y_train)
  y_test = y_test.map(lambda s: '<unknown>' if s not in le.classes_ else s)
  le.classes_ = np.append(le.classes_, '<unknown>')
  y_test_enc = le.transform(y_test)
  return y_train_enc, y_test_enc, le 


# COMMAND ----------

# DBTITLE 1,RUNNING WHEN MODEL IS ALREADY PERSISTED

## KEEP IN GC
top_variables_list = []
top_vars = []

if (read_model == True):
  try:
    with open(path + model_ID.strip() +"_le.pkl", "rb") as file: #reading Label Encoder
      le = pk.load(file)
    with open(path + model_ID.strip() +".pkl", "rb") as file: #reading model
      model = pk.load(file)
  except:
    raise Exception("Model not persisted")
  
  top_vars = []
  top_variables = spark.sql(("select * from data_science.top_vars where model_id = '{model_ID}'").format(model_ID = model_ID)).toPandas() #reading top features
  top_variables_value = top_variables['list'].iloc[0] 
  top_variables_value = top_variables_value[1:-1] # This is to remove the parenthesis
  top_variables_list = top_variables_value.split(",")
  for i in top_variables_list:
    i = i.strip()
    i = i[1:-1] # This is to remove the apostrophies in predictor names
    top_vars.append(i)
  print(top_vars)
  if('target' in pred_df.columns):
    pred_df = pred_df.drop(['target'],axis=1)
  if(target_name in pred_df.columns):
    pred_df = pred_df.drop([target_name],axis=1)
  print(pred_df.shape)
  # pred_df.head()

  ## added in preprocessor
  # try:
  #   print(pred_df.shape)
  #   pred_df_pp = feature_encoding(pred_df, "validation")
  #   pred_df_pp = pred_df_pp.filter(top_vars)
  #   for i in top_vars: #test data predictors should be consistent with train data predictors.(done to take care of any extra or missing category in categorical predictors)
  #     if i not in pred_df_pp.columns:
  #       pred_df_pp[i] = 0
  #   print(pred_df_pp.columns)
  #   display(pred_df_pp.head())
  #   imp = SimpleImputer(strategy="most_frequent")
  #   pred_idf=pd.DataFrame(imp.fit_transform(pred_df_pp))
  #   pred_idf.columns=pred_df_pp.columns
  #   pred_idf.index=pred_df_pp.index
  #   print(pred_idf.shape)
    
  # except Exception as e:
  #   logging.error(e)
  #   logging.shutdown()
  #   raise Exception("FE error")

  predictions = predict_target(model, pred_df)

  if(show_probability == True):
    pred_overall_df =  predict_target_prob(model, pred_df, class_names=le.inverse_transform(model.classes_))
    
  pred_transformed = le.inverse_transform(predictions)
  pred_df[target_name] = pred_transformed

# COMMAND ----------

## KEEP IN GC
if (read_model == True):
  if('id' not in pred_df_orig.columns):
    print("adding id to pred df orig")
    pred_df_orig['id'] = pred_df_orig[id_name]

  pred_df_all = pd.concat([pred_df.reset_index(drop=True),pred_overall_df.reset_index(drop=True)],axis=1)

  if (target_name + "_actual" in pred_df_all.columns):
    pred_df_all = pred_df_all.drop(columns = {target_name + "_actual"})
    
  pred_df_all = pred_df_all.merge(pred_df_orig.filter(['id',target_name+'_actual']),how='inner',on=['id'])
  print(pred_df_all.shape)
  pred_df_all.head()

  try:
    final_df = pred_df_orig.drop(target_name,axis=1)
    final_df = final_df.merge(pred_df.filter(['id', target_name, target_name + "_pref"]),on='id',how='inner')
    if((id_name!='id')&('id' in final_df.columns)):
      final_df = final_df.drop(['id'],axis=1)
      
  except Exception as e:
    logging.error(e)
    logging.shutdown()
    raise Exception("Sp error")
    
  #OUTPUT PERSISTENCE
  spark.sql("drop table IF EXISTS "+ prediction_table_name+"_gc_results")
  pred_df_all_persist = pred_df_all.copy()

  pred_df_all_persist.columns = ["pred_"+str(i) if str(i)[0] in ['1','2','3','4','5','6','7','8','9','0'] else str(i) for i in pred_df_all_persist.columns ]
  pred_df_all_persist['model_id'] = model_ID
  pred_df_all_persist_spark = spark.createDataFrame(pred_df_all_persist)

  if(output_to == "redshift"):
    try:
      persist_data(pred_df_all_persist_spark, prediction_table_name+"_gc_results", ['model_id','rid'],is_rs = True)
    except Exception as e:
      print(e)
  if(output_to == "spark"):
    try:
      persist_data(pred_df_all_persist_spark, prediction_table_name+"_gc_results", ['model_id','rid'],is_rs = False)
    except Exception as e:
      print(e)
  dbutils.notebook.exit("Run successful")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data pre-processing

# COMMAND ----------

# DBTITLE 1,1. Train Validation Split
try:
  X = input_df.drop(['target'], axis=1)
  y = input_df[['target']]
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=1,stratify=y)
  print('Train data shape::', X_train.shape, y_train.shape)
  print('Val data shape::', X_val.shape, y_val.shape)
  del X
  del y
except Exception as e:
  logging.error(e)
  logging.shutdown()
  raise Exception("TV error")
  

# COMMAND ----------

print("### TRAIN DATA ###")
print(y_train.target.value_counts(normalize=True,dropna=False))
print("### TEST DATA ###")
print(y_val.target.value_counts(normalize=True,dropna=False))

print(y_train.target.dtype)
print(y_val.target.dtype)

# COMMAND ----------

# DBTITLE 1,1. Data Shape - Train Validation Split
try:
  data_shape = {'train_size': [X_train.shape[0], X_train.shape[1]],
          'val_size': [X_val.shape[0], X_val.shape[1]]
          }
  display(pd.DataFrame(data_shape ))
except Exception as e:
  logging.error(e)
  logging.shutdown()
  raise Exception("Split error")

# COMMAND ----------

# DBTITLE 1,Train and Validation Encoding
try:
  print("X_train. ",X_train.shape)
  print("X_val. ",X_val.shape)
  # X_train = feature_encoding(X_train, "train")
  # X_val = feature_encoding(X_val, "validation")

  y_train_en, y_val_en, le = target_encoding(y_train['target'], y_val['target'])
  print('X & y train after align::', X_train.shape, y_train.shape)
  print('X & y val after align::', X_val.shape, y_val.shape)
except Exception as e:
  logging.error(e)
  logging.shutdown()
  raise Exception("Encoding error")
  

# COMMAND ----------

print(le.classes_)
print(le.transform(le.classes_))
print(le.inverse_transform(le.transform(le.classes_)))

# COMMAND ----------

X_train.head()

# COMMAND ----------

# DBTITLE 1,2. Data Shape after Encoding
try:
  data_shape = {'train_size': [X_train.shape[0], X_train.shape[1]],
          'val_size': [X_val.shape[0], X_val.shape[1]]
          }
  display(pd.DataFrame(data_shape ))
except Exception as e:
  logging.error(e)
  logging.shutdown()
  raise Exception("Encoding error")

# COMMAND ----------

top_vars=list(X_train.columns)
if 'id' in top_vars:
  top_vars.remove('id')
if id_name in top_vars:
  top_vars.remove(id_name)
print(top_vars)

# COMMAND ----------

if(persist_model == True):
  vars_df = pd.DataFrame(columns = ['model_id','list','inserted_at'])
  vars_df = vars_df.append({'model_id':model_ID,'list':str(top_vars),'inserted_at':dt.datetime.today()},ignore_index = True)
  vars_df_persist_spark = spark.createDataFrame(vars_df)
  persist_data(vars_df_persist_spark, "data_science.top_vars", ['model_id'],is_rs = False)

# COMMAND ----------

print("Train data shape : ",X_train.shape)
print("Val data shape : ",X_val.shape)
print(len(y_train_en))
print(len(y_val_en))

# COMMAND ----------

# DBTITLE 1,4. Sampling generic Function
def sample_data (sampler_type, X_data, y_data):
  y = np.squeeze(y_data)
  # Copy the dataframe without the predicted column
  temp_dataframe = X_data
  x_sampled, y_sampled = sampler_type.fit_resample(temp_dataframe, y)
  result = pd.DataFrame(x_sampled)
  # Restore the column names
  result.columns = temp_dataframe.columns
  # Restore the y values
  y_sampled = pd.Series(y_sampled)
  result['target'] = y_sampled
  del temp_dataframe
  del y
  del x_sampled
  del y_sampled
  return result

# COMMAND ----------

unique_target_values, counts_target_values = np.unique(y_train_en, return_counts=True)
print(np.asarray((le.inverse_transform(unique_target_values), counts_target_values)).T)

# COMMAND ----------

# DBTITLE 1,4.a Under Sampling
sampling_names = ['NA']
try:
  
  under_sampling = False
  print(X_train.shape)
  input_us_df_tmp = sample_data (RandomUnderSampler(random_state=1), X_train, y_train_en)
  print(input_us_df_tmp.shape)
  vc_us = pd.DataFrame(input_us_df_tmp['target'].value_counts()).reset_index()
  vc_us['index'] = le.inverse_transform(vc_us['index'])
  print("under_sampling_criteria_count  ::  ",under_sampling_criteria_count)
  if((input_us_df_tmp.shape[0] > under_sampling_criteria_count) & (class_imbalance)):
    print("SETTING UNDER SAMPLE = TRUE")
    under_sampling = True
    sampling_names.append('us')
  # display(vc_us)
  
  print("sampling_names  ",sampling_names)
except Exception as e:
  logging.error(e)
  logging.shutdown()
  raise Exception("US  error")
 

# COMMAND ----------

if(class_imbalance):
  sampling_names.append('smote')
  sampling_names.append('os')
print(sampling_names)

# COMMAND ----------

if preferred_sampling != "":
  sampling_names = [preferred_sampling]

# COMMAND ----------

# DBTITLE 1,Conclusion
displayHTML("Feature Engineering Done")

# COMMAND ----------

# if is_update_model_param = True , it will always retrain a new model with mode_combos as model_names * sampling_names , hyper_parameters_persisted = {}
# if is_update_model_param = False , it will search if there exists a record for stored model params in data_science.gc_hyper_parameters , if yes take it , if not behave like 1st option
if(is_update_model_param):
  print("inside is_update_model_param=True")
  hyper_parameters_persisted={}
  try:
  ## Create permutations of model combos
    model_combos = []
    for model in model_names:
      for sampling in sampling_names:
        model_combo = str(model +"_" + sampling)
        model_combos.append(model_combo)
    
  except Exception as e:
    logging.error(e)
    logging.shutdown() 
    raise Exception("MC error")
else:
  print("inside is_update_model_param=false")
  df_hp_stored=spark.sql("select * from data_science.gc_hyper_parameters where model_ID='"+model_ID+"'").toPandas()
  if(df_hp_stored.shape[0]!=0):
    model_combos=[df_hp_stored['model_name'].iloc[0]]
    hyper_parameters_persisted=dict(json.loads(df_hp_stored['hyper_parameters'].iloc[0]))
    print("model_combos  ",model_combos)
    print("hyper_parameters_persisted  ",hyper_parameters_persisted)
  else:
    hyper_parameters_persisted={}
    model_combos = []
    for model in model_names:
      for sampling in sampling_names:
        model_combo = str(model +"_" + sampling)
        model_combos.append(model_combo)
print("model_combos  ::  ",model_combos)

# COMMAND ----------

# MAGIC %md
# MAGIC ## GRID SERACH

# COMMAND ----------

import random
from sklearn.model_selection import cross_val_score

gbl = globals()
rep_id_count=1  

initial_run_check=True

for i in model_combos:
    no_random_grid_search_tries=1
    print(i)
    model_name = i.split('_')[0]
    sampling_type = i.split('_')[1]

    df_tmp=X_train
    if(id_name in df_tmp.columns):
      df_tmp=df_tmp.drop(id_name,axis=1)
    if('id' in df_tmp.columns):
      df_tmp=df_tmp.drop('id',axis=1)
      
    df_tmp['target']=y_train_en
    df_spark=spark.createDataFrame(df_tmp)
    df_spark=df_spark.withColumn("model_name",F.lit(str(model_name)))
    df_spark=df_spark.withColumn("sampling_type",F.lit(str(sampling_type)))
    
    if((is_gridsearch) & (model_name in ('rf','xg'))):
      if(model_name=='xg'):
        no_random_grid_search_tries=40
      elif(model_name=='rf'):
        no_random_grid_search_tries=20
        
      replication_df = spark.createDataFrame(pd.DataFrame(list(range(rep_id_count,
                                                                     rep_id_count+no_random_grid_search_tries)),
                                                          columns=['replication_id']))
      rep_id_count = rep_id_count + no_random_grid_search_tries
    else:
      replication_df = spark.createDataFrame(pd.DataFrame(list(range(rep_id_count,
                                                                     rep_id_count+1)),
                                                          columns=['replication_id']))
      rep_id_count = rep_id_count + 1
      
      
    if(initial_run_check):
      replicated_train_df = df_spark.crossJoin(replication_df)
      initial_run_check=False
    else:
      replicated_train_df=replicated_train_df.union(df_spark.crossJoin(replication_df))

# COMMAND ----------

def generate_model_hp(model_name,is_gridsearch=False,params_input={}):
  if(model_name=='rf'):
    if(is_gridsearch):
      # Get randomized hyperparam values
      n_estimators_hp =   random.choice(list(range(100,401)))
      max_depth_hp = random.choice(list(range(2,12)))
      max_features_hp = random.choice(['log2', 'sqrt'])
      criterion_hp = random.choice(['gini','entropy'])
      min_samples_leaf_hp = random.choice(list(range(5,15)))
      min_samples_split_hp =random.choice(list(range(5,15)))

      params=  dict({'n_estimators': n_estimators_hp,
                        'max_depth': max_depth_hp,
                        'min_samples_leaf': min_samples_leaf_hp,
                        'min_samples_split': min_samples_split_hp,
                        'max_features': max_features_hp,
                        'criterion': criterion_hp
                    })
      model = RandomForestClassifier(n_jobs=1,random_state=1)
      for k,v in params.items():
         setattr(model,k,v)
    else:
      model=RandomForestClassifier(n_jobs=1,random_state=1)
      params=params_input
      print(params)
      for k,v in params.items():
        setattr(model,k,v)

  
  elif(model_name == "dt"):
    model = DecisionTreeClassifier(random_state=1)
    params=params_input
    for k,v in params.items():
      setattr(model,k,v)
    
  elif(model_name == "xg"):
    # obj="multi:softprob"
    obj = "multi:softmax"
    if(is_gridsearch):
      learning_rate_hp = random.choice([0.001, 0.01])
      n_estimators_hp = random.choice([100,300,500])
      max_depth_hp = random.choice(list(range(3,9,2))) # changes from (3,13,3)
      gamma_hp = random.choice([0,0.01,0.05, 0.1,0.2,0.3])
      colsample_bytree_hp = random.choice([0.3,0.5,0.7])
      subsample_hp = random.choice([0.5,0.7])
      reg_alpha_hp = random.choice([0, 1e-2, 0.1,0.5])
      reg_lambda_hp = random.choice([ 0.1, 1,5, 10, 100])
      min_child_weight_hp = random.choice([1,3,5,10,15])
      

      params=  dict({'learning_rate': learning_rate_hp,
                     'n_estimators':n_estimators_hp,
                     'max_depth':max_depth_hp,
                     'gamma':gamma_hp,
                     'colsample_bytree':colsample_bytree_hp,
                     'reg_alpha':reg_alpha_hp,
                     'reg_lambda':reg_lambda_hp,
                     'min_child_weight':min_child_weight_hp,
                     'subsample':subsample_hp
                    })
      model = xgb.XGBClassifier(n_jobs=1,objective=obj, random_state=1, num_class = target_classes,verbosity=1,
                                tree_method = 'hist',
                                use_label_encoder=False)

      for k,v in params.items():
        setattr(model,k,v)
    else:
      model = xgb.XGBClassifier(n_jobs=1,objective=obj, random_state=1, num_class = target_classes,verbosity=1,
                                tree_method = 'hist',
                                use_label_encoder=False)
      params=params_input
      for k,v in params.items():
        setattr(model,k,v)

  elif(model_name == "nb"):
    model = GaussianNB()
    params=params_input
    for k,v in params.items():
      setattr(model,k,v)

  elif(model_name == "ovr"):
    #Tried SVC, DT, RF --> TRY LG as well and whats the diff bet soft amx and this for XG
    model = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3),n_jobs=1)
    params=params_input
    for k,v in params.items():
      setattr(model,k,v)
    
  return model,params

# COMMAND ----------

from imblearn.pipeline import Pipeline, make_pipeline

outSchema = StructType([StructField('replication_id',IntegerType(),True),
                        StructField('model_name',StringType(),True),
                        StructField('sampling_type',StringType(),True),
                        StructField('cv_score',DoubleType(),True),
                        StructField('hyper_parameters',StringType(),True)
                       ])

from sklearn.metrics import fbeta_score, make_scorer

if(accuracy_focus=='precision'):
  scorer = make_scorer(fbeta_score,average='macro', beta=0.5)
elif(accuracy_focus=='recall'):
  scorer = make_scorer(fbeta_score,average='macro', beta=2)
elif(accuracy_focus=='accuracy'):
  scorer = accuracy_focus
else:
  scorer = make_scorer(fbeta_score,average='macro', beta=1)

def run_model(pdf):
    replication_id = pdf['replication_id'].iloc[0]
    model_name = pdf['model_name'].iloc[0] 
    sampling_type = pdf['sampling_type'].iloc[0]
    
    print("replication_id  ",replication_id)
    print("model_name  ",model_name)
    print("sampling_type  ",sampling_type)

    if((len(hyper_parameters_persisted.keys())>0)&(is_update_model_param==False)):
      model,params = generate_model_hp(model_name,is_gridsearch,params_input=hyper_parameters_persisted)
    else:
      model,params = generate_model_hp(model_name,is_gridsearch)
    
    if( 'replication_id' in pdf.columns):
      pdf=pdf.drop('replication_id',axis=1)
    if( 'model_name' in pdf.columns):
      pdf=pdf.drop('model_name',axis=1)
    if( 'sampling_type' in pdf.columns):
      pdf=pdf.drop('sampling_type',axis=1)
    if('id' in pdf.columns):
      pdf=pdf.drop('id',axis=1)
    if(id_name in pdf.columns):
      pdf=pdf.drop(id_name,axis=1)
    if(target_name+'_actual' in pdf.columns):
      pdf=pdf.drop(target_name+'_actual',axis=1)
      
    X = pdf.drop(['target'], axis=1).values  
    y = pdf[['target']].values.ravel()
    
    if(sampling_type=='us'):
      imba_pipeline = make_pipeline(RandomUnderSampler(random_state=1), 
                              model)
    elif(sampling_type=='smote'):
      imba_pipeline = make_pipeline(SMOTE(random_state=1), 
                              model)
    elif(sampling_type=='os'):
      imba_pipeline = make_pipeline(RandomOverSampler(random_state=1), 
                              model)
    else:
      imba_pipeline = make_pipeline(model)
    
    score = cross_val_score(imba_pipeline, X, y, scoring=scorer, cv=cv_kfold ,verbose=2,n_jobs=-1)

    # 5. return results as pandas DF
    hp_str=json.dumps(params)
    res =pd.DataFrame({'replication_id':replication_id,
                       'model_name':model_name,
                       'sampling_type':sampling_type,
                       'cv_score':score,
                       'hyper_parameters':hp_str
                      })
    
    return res

# COMMAND ----------

results = replicated_train_df.groupby("replication_id").applyInPandas(run_model,outSchema)
# results.persist()

# COMMAND ----------

results_pd = results.toPandas()
results_pd.head(10)

# COMMAND ----------

results_pd_agg = results_pd.groupby(['replication_id','model_name','sampling_type','hyper_parameters']).agg({'cv_score':['mean','std']})
results_pd_agg.columns = ['_'.join(x).strip() for x in results_pd_agg.columns]
results_pd_agg = results_pd_agg.reset_index()
results_pd_agg = results_pd_agg.sort_values(by=['cv_score_mean'],ascending=False)
display(results_pd_agg)

# COMMAND ----------

cv_results = results_pd_agg.groupby(['model_name','sampling_type']).head(1)
display(cv_results)

if(cv_score_persistence_table != 'false'):
  cv_results_persist = cv_results.copy()
  cv_results_persist['model_id'] = model_ID
  table_name = cv_score_persistence_table
  cv_results_spark = spark.createDataFrame(cv_results_persist)
  persist_data(cv_results_spark, table_name, ['model_id','model_name','sampling_type'],is_rs = False)


# COMMAND ----------

cv_results = cv_results.sort_values(by=['cv_score_mean'],ascending=False)
cv_results_best = cv_results.head(1)
best_model = cv_results_best['model_name'].iloc[0] + "_" + cv_results_best['sampling_type'].iloc[0]
best_model

# COMMAND ----------

try:
  target_lst_en = le.transform(target_lst)
  print("target_lst. ",target_lst)
  print("target_lst_en. ",target_lst_en)  
except Exception as e:
  logging.error(e)
  logging.shutdown()
  raise Exception("T/F error")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model building and Prediction of validation data

# COMMAND ----------

def get_percentile_buckets(df,all_positive=False):
  df_tmp = df.copy()
  df_tmp['pred'] = df_tmp.idxmax(axis=1)
  df_tmp['actual'] = le.inverse_transform(y_val_en)
  df_tmp['model_id'] = model_ID

  def generate_actual_vs_predicted_report(df):
    y_val = df['actual'].astype(str)
    y_val_pred = df['pred'].astype(str)
    report = classification_report(y_val, y_val_pred, output_dict=True)
    cl_rep = round((pd.DataFrame(report).transpose().reset_index()),4)
    cl_rep = cl_rep.rename(columns={'index': 'value'})
    return cl_rep

  def percentile_breakdown(df):
    output_df_prob_percentiles = pd.DataFrame(columns =['value','precision','recall','f1-score','support','percentile'],dtype=object)
    bin_labels = [.99,.95,.90,.75,.50,.25,0]
    for i in bin_labels:
      n_th_percentile = df[target_lst].quantile(i).values[0]
      df_prob_x = df[df[target_lst[0]] >= n_th_percentile].copy()
      if all_positive:
        df_prob_x['pred'] = target_lst[0]
      print(df_prob_x.shape)
      row = generate_actual_vs_predicted_report(df_prob_x)
      row['percentile'] = round((1-i)*100)
      output_df_prob_percentiles = output_df_prob_percentiles.append(row)
    return output_df_prob_percentiles

  cl_rep = df_tmp.groupby(['model_id'],as_index=False).apply(percentile_breakdown)
  return cl_rep

# COMMAND ----------

X_train.loc[0]

# COMMAND ----------

print(X_train.shape)
X_train.describe()

# COMMAND ----------

pd.Series(y_train_en).value_counts()

# COMMAND ----------

# DBTITLE 1,Model training
from sklearn.utils import parallel_backend
import sklearn
from sklearn.metrics import fbeta_score
from joblibspark import register_spark
register_spark()

# try:
validation_df = pd.DataFrame(columns=['model_name','recall', 'precision', 'f1','f_beta',
                                      'overall_recall', 'overall_precision', 'overall_f1','overall_fbeta', 'accuracy'],dtype=object)

gbl = globals()

for i in model_combos:
  print(i)
  model_name = i.split('_')[0]
  sampling_type = i.split('_')[1]
  print(model_name)
  print(sampling_type)

  if(id_name in X_train.columns):
    X_train=X_train.drop(id_name,axis=1)
  if('id' in X_train.columns):
    X_train=X_train.drop('id',axis=1)
  
  if(sampling_type=='us'):
    df = sample_data (RandomUnderSampler(random_state=1), X_train, y_train_en)
  elif(sampling_type=='smote'):
    df = sample_data (SMOTE(random_state=1), X_train, y_train_en)
  elif(sampling_type=='os'):
    df = sample_data (RandomOverSampler(random_state=1), X_train, y_train_en)
  else:
    df = X_train
    df['target'] = y_train_en
  
  if(id_name in df.columns):
    df=df.drop(id_name,axis=1)
  if('id' in df.columns):
    df=df.drop('id',axis=1)
  if(target_name+'_actual' in df.columns):
    df=df.drop(target_name+'_actual',axis=1)
    
  hyper_parameters_i = dict(json.loads(cv_results.query("model_name==@model_name and  \
                          sampling_type==@sampling_type")['hyper_parameters'].iloc[0]))

  try:
    mlflow.end_run()
    mlflow.start_run(experiment_id=exp_id,run_name=i)
    mlflow.autolog()
    mlflow.log_param("best_model",best_model)
  except Exception as e:
    print(e)
        
  model,params = generate_model_hp(model_name,is_gridsearch=False,params_input=hyper_parameters_i)

  if(model_name in ('rf','xg')):
    print("setting n_jobs = -1 inside xgboost and rf")
    setattr(model,'n_jobs',-1)
  
  if('model_name' in df.columns):
    df = df.drop('model_name',axis=1)
  if('sampling_type' in df.columns):
    df = df.drop('sampling_type',axis=1)
    
  X_t_columns = df.drop(['target'], axis=1).columns
  X = df.drop(['target'], axis=1).values  
  y = df[['target']].values.ravel()
  if 'id' in X_val.columns:
    X_val_df = X_val.drop('id', axis =1).values 
  with parallel_backend('spark',n_jobs=-1):
    if model_name == 'xg':
      model_fit = model.fit(X,y,eval_metric="mlogloss",eval_set=[(X, y), (X_val_df, y_val_en)])
      try:
        results = model.evals_result()
        fig_loss = plt.figure()
        plt.plot(results["validation_0"]["mlogloss"], label="Training loss")
        plt.plot(results["validation_1"]["mlogloss"], label="Validation loss")
        plt.axvline(int(model.best_ntree_limit), color="gray", label="Optimal tree number")
        plt.title("XgBoost - Model Multi-class Log Loss")
        plt.xlabel("Number of trees")
        plt.ylabel("Loss")
        plt.legend()
        mlflow.log_figure(fig_loss,"xg_model_loss.png")
      except Exception as e:
        print("ERROR in plotting the Training vs Validation Loss")
        print(e)
    else:
      model_fit = model.fit(X,y)

  try:
    ft_imp = model.feature_importances_
    ft_imp = pd.DataFrame(ft_imp, index=X_t_columns, columns=["importance"])
    ft_imp['variable_name'] = ft_imp.index
    ft_imp = ft_imp.sort_values(['importance'], ascending = False).filter(['variable_name', 'importance'])
    ft_imp.to_csv(model_name+"_feature_importance_self.csv",index=False)
    mlflow.log_artifact(model_name+"_feature_importance_self.csv")
  except Exception as e:
    print("ERROR in finding feature_importance")
    print(e)

  y_pred = model.predict(X)

  acc_train = round(accuracy_score(y, y_pred),2)
  prec_train_overall = round(precision_score(y, y_pred,average='macro'),2)
  recall_train_overall = round(recall_score(y, y_pred,average='macro'),2)
  f1_train_overall = round(f1_score(y, y_pred,average='macro'),2)
  if(accuracy_focus == 'precision'):
    overall_fbeta_train = fbeta_score(y, y_pred,beta=0.5,average='macro')
  elif(accuracy_focus == 'recall'):
    overall_fbeta_train = fbeta_score(y, y_pred,beta=2,average='macro')
  else:
    overall_fbeta_train = fbeta_score(y, y_pred,beta=1,average='macro')

  if(accuracy_focus == 'precision'):
    fbeta_train = fbeta_score(y, y_pred, labels=target_lst_en,beta=0.5,average='macro')
  elif(accuracy_focus == 'recall'):
    fbeta_train = fbeta_score(y, y_pred, labels=target_lst_en,beta=2,average='macro')
  else:
    fbeta_train = fbeta_score(y, y_pred, labels=target_lst_en,beta=1,average='macro')

  prec_train = round(precision_score(y, y_pred,labels=target_lst_en,average='macro'),2)
  recall_train = round(recall_score(y, y_pred,labels=target_lst_en,average='macro'),2)
  f1_train = round(f1_score(y, y_pred,labels=target_lst_en,average='macro'),2)
  
  if len(unique_target_values) < 3:
    auc = round(metrics.roc_auc_score(y, y_pred),2)
    if model.classes_[1] == int(target_lst_en[0]):
      y_pred_prob = model.predict_proba(X)[:, 1]
    else:
      y_pred_prob = model.predict_proba(X)[:, 0]

    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred_prob)

    fig = plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Training')
    plt.legend(loc="lower right")
    mlflow.log_figure(fig,"training_roc_curve_self.png")

    fig_rp = plt.figure()
    plt.plot(metrics.precision_recall_curve(y,y_pred_prob)[1], metrics.precision_recall_curve(y,y_pred_prob)[0], color='darkorange', lw=2, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Training')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    mlflow.log_figure(fig_rp,"training_precision_recall_curve_self.png")
  else:
    print(y.shape)
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y)
    print(y_bin.shape)
    y_pred_prob = model.predict_proba(X)
    auc = round(metrics.roc_auc_score(y_bin, y_pred_prob, multi_class='ovo'),2) #'ovo' over 'ovr' because it is insensitive to class imbalance

  class_names = np.unique(le.inverse_transform(y))
  cm_display = plot_confusion_matrix(model,X,y,
                        cmap=plt.cm.Blues, 
                        display_labels=class_names, normalize = 'true')
  fig, ax = plt.subplots()
  cm_display.plot(ax=ax)
  mlflow.log_figure(fig,"train_confusion_matrix_normalized_self.png")

  cm_display_1 = plot_confusion_matrix(model, X, y ,
                        cmap=plt.cm.Blues, 
                        display_labels=class_names)
  fig_1, ax_1 = plt.subplots()
  cm_display_1.plot(ax=ax_1)
  mlflow.log_figure(fig_1,"train_confusion_matrix_self.png")

  ####################################### VALIDATION DF ############################################
  
  gbl[i+'_model'] = model_fit
  gbl[i+'_pred'] = predict_target(gbl[i+'_model'], X_val)
  gbl[i+'_pred_prob'] = predict_target_prob(gbl[i+'_model'], X_val, class_names=le.inverse_transform(gbl[i+'_model'].classes_))

  y_pred = gbl[i+'_pred']

  recall = recall_score(y_val_en, y_pred, labels=target_lst_en, average='macro')
  precision = precision_score(y_val_en, y_pred, labels=target_lst_en, average='macro')
  f1 = f1_score(y_val_en, y_pred, labels=target_lst_en, average='macro')
  if(accuracy_focus=='precision'):
    fbeta=fbeta_score(y_val_en, y_pred, labels=target_lst_en,beta=0.5,average='macro')
  elif(accuracy_focus=='recall'):
    fbeta=fbeta_score(y_val_en, y_pred, labels=target_lst_en,beta=2,average='macro')
  else:
    fbeta=fbeta_score(y_val_en, y_pred, labels=target_lst_en,beta=1,average='macro')
      
  overall_recall = recall_score(y_val_en, y_pred,  average='macro')
  overall_precision = precision_score(y_val_en, y_pred, average='macro')
  overall_f1 = f1_score(y_val_en, y_pred, average='macro')
  overall_accuracy = accuracy_score(y_val_en, y_pred)
  
  if(accuracy_focus=='precision'):
    overall_fbeta=fbeta_score(y_val_en, y_pred,beta=0.5,average='macro')
  elif(accuracy_focus=='recall'):
    overall_fbeta=fbeta_score(y_val_en, y_pred,beta=2,average='macro')
  else:
    overall_fbeta=fbeta_score(y_val_en, y_pred,beta=1,average='macro')
    
  overall_fbeta=fbeta_score(y_val_en, y_pred,beta=0.5,average='macro')

  validation_df = validation_df.append({'model_name' : i , 'recall' : recall, 'precision' : precision , 'f1' :f1, 'f_beta':fbeta, 'overall_recall':overall_recall, 'overall_precision' : overall_precision, 'overall_f1' : overall_f1,'overall_fbeta':overall_fbeta, 'accuracy':overall_accuracy}, ignore_index=True)

  if len(unique_target_values) < 3:
    auc_val = round(metrics.roc_auc_score(y_val_en, y_pred),2)
    if model.classes_[1] == int(target_lst_en[0]):
      y_pred_prob = gbl[i+'_pred_prob'].iloc[:, 1]
    else:
      y_pred_prob = gbl[i+'_pred_prob'].iloc[:, 0]

    fpr_val, tpr_val, thresholds_val = metrics.roc_curve(y_val_en, y_pred_prob)

    fig_val = plt.figure()
    plt.plot(fpr_val, tpr_val, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % auc_val)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Validation')
    plt.legend(loc="lower right")
    mlflow.log_figure(fig_val,"validation_roc_curve_self.png")

    fig_rp_val = plt.figure()
    plt.plot(metrics.precision_recall_curve(y_val_en,y_pred_prob)[1], metrics.precision_recall_curve(y_val_en,y_pred_prob)[0], color='darkorange', lw=2, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Validation')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    mlflow.log_figure(fig_rp_val,"validation_precision_recall_curve_self.png")
  else:
    y_bin_val = lb.fit_transform(y_val_en)
    print(y_bin_val.shape)
    y_pred_prob = gbl[i+'_pred_prob']
    print(y_pred_prob.shape)
    auc_val = round(metrics.roc_auc_score(y_bin_val, y_pred_prob, multi_class='ovo'),2)

  class_names = np.unique(le.inverse_transform(y_val_en))
  X_val_df = X_val.drop('id', axis =1).values 
  cm_display = plot_confusion_matrix(model, X_val_df, y_val_en, cmap=plt.cm.Blues, display_labels=class_names, normalize ='true')
  fig, ax = plt.subplots()
  cm_display.plot(ax=ax)
  mlflow.log_figure(fig,"val_confusion_matrix_normalized_self.png")
  cm_display_1 = plot_confusion_matrix(model, X_val_df, y_val_en, cmap=plt.cm.Blues, display_labels=class_names)
  fig_1, ax_1 = plt.subplots()
  cm_display_1.plot(ax=ax_1)
  mlflow.log_figure(fig_1,"val_confusion_matrix_self.png")

  clsf_report = pd.DataFrame(metrics.classification_report(y_val_en, y_pred, output_dict=True)).round(4).transpose()
  clsf_report.insert(loc=0, column='class', value=list(set(y_val_en)) + ["accuracy", "macro avg", "weighted avg"])
  clsf_report.to_csv(model_name+'_clf_report.csv', index= False)

  metrics_df = pd.DataFrame([["training_accuracy",acc_train],["training_precision",prec_train],["training_recall",recall_train],["training_fbeta",fbeta_train],["training_f1",f1_train],["training_auc_roc",auc],["training_precision_overall",prec_train_overall],["training_recall_overall",recall_train_overall],["training_fbeta_overall",overall_fbeta_train],["training_f1_overall",f1_train_overall],['val_recall',recall], ['val_precision',precision], ['val_f1',f1], ["val_auc_roc",auc_val], ['val_f_beta',fbeta], ['val_overall_recall',overall_recall],['val_overall_precision',overall_precision],['val_overall_f1',overall_f1],['val_overall_fbeta',overall_fbeta], ['val_accuracy',overall_accuracy]], columns=['Metric', 'Value'])

  validation_df = round(validation_df, 4)
  metrics_df = round(metrics_df,3)
  metrics_df.to_csv(model_name+"_metrics_df.csv",index=False)

  df_tmp = gbl[i + '_pred_prob'].copy()
  percentile_bucket_df = get_percentile_buckets(df_tmp,all_positive=False)
  percentile_bucket_df.to_csv(model_name+"_percentile_buckets.csv",index=False)

  percentile_bucket_df = get_percentile_buckets(df_tmp,all_positive=True)
  percentile_bucket_df.to_csv(model_name+"_percentile_buckets_allpositive.csv",index=False)

  mlflow.log_artifact(model_name+"_metrics_df.csv")
  mlflow.log_artifact(model_name+"_clf_report.csv")
  mlflow.log_artifact(model_name+"_percentile_buckets.csv")
  mlflow.log_artifact(model_name+"_percentile_buckets_allpositive.csv")

  if(accuracy_focus=='accuracy'):
    validation_df = validation_df.sort_values("accuracy", ascending=False)
  else:
    validation_df = validation_df.sort_values("f_beta", ascending=False)
  try:
    mlflow.end_run()
  except:
    print("exception in ending mlflow")

# except Exception as e:
#   logging.error(e)
#   logging.shutdown()
#   raise Exception("MV error")

# COMMAND ----------

def get_top_val(ensemble_df):
    return ensemble_df.value_counts().idxmax()

# COMMAND ----------

# DBTITLE 1,D. Model Evaluation Analysis
displayHTML("1. Cross Validate across all the models"
            +"<br><br>2. Model Selection"
           +"<br><br>3. Classification Validation Report")

# COMMAND ----------

# DBTITLE 1,1. Validation Results
display(validation_df)

# COMMAND ----------

import datetime

if is_update_model_param:

    best_model_df = validation_df[validation_df["model_name"] == best_model]
    model_name = best_model_df["model_name"].iloc[0].split("_")[0]
    sampling_type = best_model_df["model_name"].iloc[0].split("_")[1]
    print("model_name ", model_name)
    print("sampling_type  ", sampling_type)
    hyper_parameters_best_model = dict(
        json.loads(
            cv_results.query(
                "model_name==@model_name and  \
                             sampling_type==@sampling_type"
            )["hyper_parameters"].iloc[0]
        )
    )
    best_model_df["model_ID"] = model_ID
    best_model_df["hyper_parameters"] = json.dumps(hyper_parameters_best_model)
    best_model_df["historic_table_name"] = historic_table_name
    best_model_df["prediction_table_name"] = prediction_table_name
    best_model_df["is_gridsearch"] = is_gridsearch
    best_model_df["target_name"] = target_name
    best_model_df["target_focus"] = target_focus
    best_model_df["preferred_model"] = preferred_model
    best_model_df["inserted_at"] = pd.to_datetime(datetime.datetime.now()).strftime("%d/%m/%Y %H:%M:%S")
    print(best_model_df.shape)
    best_model_df = best_model_df.filter(
        [
            "model_ID",
            "model_name",
            "hyper_parameters",
            "historic_table_name",
            "prediction_table_name",
            "is_gridsearch",
            "target_name",
            "target_focus",
            "preferred_model",
            "recall",
            "precision",
            "f1",
            "f_beta",
            "overall_recall",
            "overall_precision",
            "overall_f1",
            "overall_fbeta",
            "accuracy",
            "inserted_at",
        ]
    )
    print(best_model_df.shape)
    #   best_model_df
    print(
        "deleting the record from data_science.gc_hyper_parameters if exists and persisting the updated record "
    )
    spark.sql(
        "delete from data_science.gc_hyper_parameters where model_ID=='"
        + model_ID
        + "'"
    )
    spark.createDataFrame(best_model_df).write.insertInto(
        "data_science.gc_hyper_parameters"
    )

# COMMAND ----------

# DBTITLE 1,2. Model Selection 
try:
  best_model_name = best_model
  print(best_model_name)
  y_val_pred = le.inverse_transform(gbl[best_model_name+'_pred'])
  gbl[best_model_name+'_cm'] = pd.DataFrame(confusion_matrix(y_val, y_val_pred))
  displayHTML("Model Chosen::   " + best_model_name)
  print(gbl[best_model_name+'_cm'])
except Exception as e:
  logging.error(e)
  logging.shutdown()
  raise Exception("MS error")

# COMMAND ----------

try:
  sm_precision = round(precision_score(y_val.target, y_val_pred, average='weighted'),4)
  sm_recall = round(recall_score(y_val.target, y_val_pred, average='weighted'),4)
  sm_f1 = round(f1_score(y_val.target, y_val_pred, average='weighted'),4)
  
  sm_precision_sel = round(precision_score(y_val.target, y_val_pred, labels=target_lst, average='weighted'),4)
  sm_recall_sel = round(recall_score(y_val.target, y_val_pred,labels=target_lst, average='weighted'),4)
  sm_f1_sel = round(f1_score(y_val.target, y_val_pred,labels=target_lst, average='weighted'),4)

except Exception as e:
  logging.error(e)
  logging.shutdown()
  raise Exception("PA error")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Precision Buckets - Validation

# COMMAND ----------

# DBTITLE 1,Precision Buckets on Validation Data
if((prec_buckets_persistence_table != 'false')&(target_focus[0] != 'all')):
  
  target_lst_en = le.transform(target_lst)
  df_tmp=gbl[best_model +'_pred_prob'].copy()
  df_tmp['pred']= df_tmp.idxmax(axis=1)
  df_tmp['actual']=le.inverse_transform(y_val_en)

  
  recall = recall_score(df_tmp['actual'], df_tmp['pred'], labels=target_lst, average='macro')
  print(recall)
  precision = precision_score(df_tmp['actual'], df_tmp['pred'], labels=target_lst, average='macro')
  print(precision)
  f1 = f1_score(df_tmp['actual'], df_tmp['pred'], labels=target_lst, average='macro')
  print(f1)
  
  output_df_prob_percentiles = pd.DataFrame(columns =['percentile','percentile_value','accuracy','precision','recall','f1_score','users'],dtype=object)
  bin_labels = [.99,.98,.97,.96,.95,.90,.85,.80,.75,.70,.65,.60,.55,.50,.45,.40,.35,.30,.25,.20,.15,.10,.05,0]

  def find_metric(df,n,nth):
    actual = df['actual']
    predicted = df['pred']
    Accuracy = round((metrics.accuracy_score(actual, predicted))*100,2)
    Precision = round((metrics.precision_score(actual, predicted,labels=target_lst,average='macro'))*100,2)
    Recall = round((metrics.recall_score(actual, predicted,labels=target_lst,average='macro'))*100,2)
    F1_score = round((metrics.f1_score(actual, predicted,labels=target_lst,average='macro'))*100,2)
    p = (1-n)*100
    p = round(p,1)
    shape = df.shape[0]
    return {'percentile':p,'percentile_value':nth,'accuracy':Accuracy,'precision':Precision,'recall':Recall,'f1_score':F1_score,'users':shape}
     
  for i in bin_labels:
    n_th_percentile = df_tmp[target_lst].quantile(i).values[0]
    output_df_prob_x = df_tmp[df_tmp[target_lst[0]] >= n_th_percentile].copy()
#     output_df_prob_x['pred']=target_lst[0]
    row = find_metric(output_df_prob_x,i,n_th_percentile)
    output_df_prob_percentiles = output_df_prob_percentiles.append(row,ignore_index = True)
    del output_df_prob_x
#   display(output_df_prob_percentiles[['percentile','precision','percentile_value']])
  
  table_name = prec_buckets_persistence_table
  output_df_prob_percentiles_persist = output_df_prob_percentiles[['percentile','precision','percentile_value','users']].copy().reset_index()
  output_df_prob_percentiles_persist['model_id'] = model_ID
  output_df_prob_percentiles_persist_spark = spark.createDataFrame(output_df_prob_percentiles_persist)
  if(prec_buckets_persistence_table!='false'):
    persist_data(output_df_prob_percentiles_persist_spark, table_name, ['model_id'],is_rs = False)
  display(output_df_prob_percentiles)

# COMMAND ----------

# DBTITLE 1,3. Classification Validation Report
def generate_actual_vs_predicted_report(df):
  m_id = df['model_id'].iloc[0]
  y_val = df[target_name +'_actual'].astype(str)
  y_val_pred = df[target_name].astype(str)
  report = classification_report(y_val, y_val_pred, output_dict=True)
  cl_rep = round((pd.DataFrame(report).transpose().reset_index()),4)
  cl_rep = cl_rep.rename(columns={'index': 'value'})
  cl_rep['model_id'] = m_id
  return cl_rep
  
def percentile_breakdown(df):
  output_df_prob_percentiles = pd.DataFrame(columns =['value','precision','recall','f1-score','support','model_id','percentile'],dtype=object)
  bin_labels = [.99,.95,.90,.75,.50,.25,0]
  for i in bin_labels:
    n_th_percentile = df[target_lst].quantile(i).values[0]
    df_prob_x = df[df[target_lst[0]] >= n_th_percentile].copy()
    print(df_prob_x.shape)
    row = generate_actual_vs_predicted_report(df_prob_x)
    row['percentile'] = round((1-i)*100)
    output_df_prob_percentiles = output_df_prob_percentiles.append(row)
  return output_df_prob_percentiles
  
try:
  target_lst_en = le.transform(target_lst)
  df_tmp = gbl[best_model +'_pred_prob'].copy()
  df_tmp[target_name]= df_tmp.idxmax(axis=1)
  df_tmp[target_name + '_actual']=le.inverse_transform(y_val_en)
  df_tmp['model_id'] = model_ID

  cl_rep = df_tmp.groupby(['model_id'],as_index=False).apply(percentile_breakdown)
  
  if(cl_report_persistence_table != 'false'):
    cl_rep_persist = cl_rep.copy()
    table_name = cl_report_persistence_table
    cl_rep_persist_spark = spark.createDataFrame(cl_rep_persist)
    persist_data(cl_rep_persist_spark, table_name, ['model_id','value'],is_rs = False)
except Exception as e:
  logging.error(e)
  logging.shutdown()
  raise Exception("VR error")

# COMMAND ----------

display(cl_rep)

# COMMAND ----------

# DBTITLE 1,E. Model Prediction 
displayHTML("1. Predict"
            +"<br><br>2. Check Distribution"
           +"<br><br>3. Persist Results")

# COMMAND ----------

# MAGIC %md
# MAGIC ## pred df

# COMMAND ----------

if('target' in pred_df.columns):
    pred_df = pred_df.drop(['target'],axis=1)
if(target_name in pred_df.columns):
  pred_df = pred_df.drop([target_name],axis=1)
print(pred_df.shape)
pred_df.head()

# COMMAND ----------

top_vars

# COMMAND ----------

# DBTITLE 1,Prediction data pre-processing
try:
  print(pred_df.shape)
  
  # pred_df_pp = feature_encoding(pred_df, "validation")
  pred_df_pp = pred_df.filter(top_vars)
  for i in top_vars: #test data predictors should be consistent with train data predictors
    if i not in pred_df_pp.columns:
      pred_df_pp[i] = 0

  imp = SimpleImputer(strategy="most_frequent")
  pred_idf = pd.DataFrame(imp.fit_transform(pred_df_pp))
  pred_idf.columns = pred_df_pp.columns
  pred_idf.index = pred_df_pp.index
  pred_idf = pred_idf.infer_objects()
  print('Before imputation :: ',pred_df_pp.shape,'\nAfter imputation :: ',pred_idf.shape)
except Exception as e:
  logging.error(e)
  logging.shutdown()
  raise Exception("FE error")

# COMMAND ----------

# DBTITLE 1,Prediction & Model Persistence
try:
  if(best_model_name != "seq_ensemble"):
    print("Not Seq Ensemble")
    m_f = gbl[best_model_name+'_model'] #best model based on validation
    if (id_name in pred_idf.columns):
      pred_idf=pred_idf.drop(id_name,axis=1)
    predictions = predict_target(m_f, pred_idf)
    if persist_model == True: #persisting the model
      with open(path + model_ID.strip() +".pkl", "wb") as file:  #DUMPING THE MODEL PICKLE FILE
        pk.dump(m_f, file)
      with open(path + model_ID.strip() +"_le.pkl", "wb") as file: #DUMPING THE LABEL ENCODER PICKLE FILE 
        pk.dump(le, file)
    if(show_probability == True):
      pred_overall_df = predict_target_prob(gbl[best_model_name+'_model'], pred_idf, class_names=le.inverse_transform(gbl[best_model_name+'_model'].classes_))
  else:
    print("Seq Ensemble")
    for i in model_combos:
      if(i != "seq_ensemble"):
        print(i)
        model_name = i.split('_')[0]
        sampling_type = i.split('_')[1]
        gbl[i+'_pred'] = predict_target(gbl[i+'_model'], pred_idf)

    ensemble_df = pd.DataFrame()
    for i in model_combos:
      ensemble_df[i+'_pred'] = gbl[i+'_pred']
    predictions = ensemble_df.max(axis=1).astype(int)
    
    if(show_probability == True):
      print("Inside Show Prob")
      class_names_pred = np.unique(le.inverse_transform(y_train_en))
      pred_overall_df = ( predict_target_prob(gbl['xg_smote'+'_model'], pred_idf, class_names_pred))

  pred_transformed = le.inverse_transform(predictions)
  pred_df[target_name] = pred_transformed

except Exception as e:
  print(e)
  logging.error(e)
  logging.shutdown()
  raise Exception("PA error")

# COMMAND ----------

if('id' not in pred_df_orig.columns):
  print("adding id to pred df orig")
  pred_df_orig['id'] = pred_df_orig[id_name]
  print(pred_df_orig.head())

# COMMAND ----------

pred_df_all = pd.concat([pred_df.reset_index(drop=True),pred_overall_df.reset_index(drop=True)],axis=1)
pred_df_all = pred_df_all.merge(pred_df_orig.filter(['id',target_name+'_actual']),how='inner',on=['id'])
print(pred_df_all.shape)
pred_df_all.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test metrics

# COMMAND ----------

def generate_actual_vs_predicted_report(df):
  y_val = df[target_name+'_actual'].astype(str)
  y_val_pred = df[target_name].astype(str)
  report = classification_report(y_val, y_val_pred, output_dict=True)
  cl_rep = round((pd.DataFrame(report).transpose().reset_index()),4)
  cl_rep = cl_rep.rename(columns={'index': 'value'})
  return cl_rep

def percentile_breakdown(df,all_positive=False):
  output_df_prob_percentiles = pd.DataFrame(columns =['value','precision','recall','f1-score','support','percentile'],dtype=object)
  bin_labels = [.99,.95,.90,.75,.50,.25,0]
  for i in bin_labels:
    n_th_percentile = df[target_lst].quantile(i).values[0]
    df_prob_x = df[df[target_lst[0]] >= n_th_percentile].copy()
    if all_positive:
      df_prob_x[target_name] = target_lst[0]
    print(df_prob_x.shape)
    row = generate_actual_vs_predicted_report(df_prob_x)
    row['percentile'] = round((1-i)*100)
    output_df_prob_percentiles = output_df_prob_percentiles.append(row)
  return output_df_prob_percentiles

ans = percentile_breakdown(pred_df_all)
ans_1 = percentile_breakdown(pred_df_all,all_positive=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Percentile wise metrics

# COMMAND ----------

display(ans)

# COMMAND ----------

display(ans_1)

# COMMAND ----------

try:
  mlflow.end_run()
  mlflow.start_run(experiment_id=exp_id,run_name='test_data')
  mlflow.autolog()
  mlflow.log_param("best_model",best_model)
except Exception as e:
  print(e)

ans.to_csv("percentile_buckets_test_data.csv",index=False)
mlflow.log_artifact("percentile_buckets_test_data.csv")

ans_1.to_csv("percentile_buckets_test_allpositive.csv",index=False)
mlflow.log_artifact("percentile_buckets_test_allpositive.csv")

# COMMAND ----------

clsf_report = pd.DataFrame(metrics.classification_report(pred_df_all[target_name +'_actual'].astype(str), pred_df_all[target_name].astype(str), output_dict=True)).round(4).transpose()
clsf_report = clsf_report.reset_index()

clsf_report.to_csv('clf_report_test_data.csv', index= False)
mlflow.log_artifact("clf_report_test_data.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Overall metrics on test data

# COMMAND ----------

display(clsf_report)

# COMMAND ----------

recall = recall_score(pred_df_all[target_name + '_actual'].astype(str), pred_df_all[target_name].astype(str), labels=target_lst_en, average='macro')
precision = precision_score(pred_df_all[target_name + '_actual'].astype(str), pred_df_all[target_name].astype(str), labels=target_lst_en, average='macro')
f1 = f1_score(pred_df_all[target_name + '_actual'].astype(str), pred_df_all[target_name].astype(str), labels=target_lst_en, average='macro')
if(accuracy_focus=='precision'):
  fbeta=fbeta_score(pred_df_all[target_name + '_actual'].astype(str), pred_df_all[target_name].astype(str), labels=target_lst_en,beta=0.5,average='macro')
elif(accuracy_focus=='recall'):
  fbeta=fbeta_score(pred_df_all[target_name + '_actual'].astype(str), pred_df_all[target_name].astype(str), labels=target_lst_en,beta=2,average='macro')
else:
  fbeta=fbeta_score(pred_df_all[target_name + '_actual'].astype(str), pred_df_all[target_name].astype(str), labels=target_lst_en,beta=1,average='macro')
    
overall_recall = recall_score(pred_df_all[target_name + '_actual'].astype(str), pred_df_all[target_name].astype(str),  average='macro')
overall_precision = precision_score(pred_df_all[target_name + '_actual'].astype(str), pred_df_all[target_name].astype(str), average='macro')
overall_f1 = f1_score(pred_df_all[target_name + '_actual'].astype(str), pred_df_all[target_name].astype(str), average='macro')
overall_accuracy = accuracy_score(pred_df_all[target_name + '_actual'].astype(str), pred_df_all[target_name].astype(str))

if len(unique_target_values) < 3:
  y_bin_test = lb.fit_transform(pred_df_all[target_name + '_actual'].astype(str))
  y_pred_prob = np.array(pred_df_all[list(le.inverse_transform(unique_target_values))].astype(float))
  print(y_bin_test.shape)
  auc = round(metrics.roc_auc_score(y_bin_test, pred_df_all[target_lst[0]].astype(float)),2)
  fpr, tpr, _ = metrics.roc_curve(y_bin_test, pred_df_all[target_lst[0]].astype(float))

  fig = plt.figure()
  plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % auc)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curve - Testing')
  mlflow.log_figure(fig,"test_roc_curve_self.png")
                    
  fig_rp = plt.figure()
  plt.plot(metrics.precision_recall_curve(y_bin_test, pred_df_all[target_lst[0]].astype(float))[1], metrics.precision_recall_curve(y_bin_test, pred_df_all[target_lst[0]].astype(float))[0], color='darkorange', marker='o')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('Precision-Recall Curve - Testing')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  mlflow.log_figure(fig_rp,"test_precision_recall_curve_self.png")
else:
  y_bin_test = lb.fit_transform(pred_df_all[target_name + '_actual'].astype(str))
  y_pred_prob = np.array(pred_df_all[list(le.inverse_transform(unique_target_values))].astype(float))
  auc = round(metrics.roc_auc_score(y_bin_test, y_pred_prob, multi_class='ovo'),2)

class_names = np.unique(pred_df_all[target_name].astype(str))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(pred_df_all[target_name + '_actual'].astype(str), pred_df_all[target_name].astype(str)), display_labels = class_names)
fig_cm, ax_cm = plt.subplots()
cm_display.plot(ax=ax_cm)
mlflow.log_figure(fig_cm,"test_confusion_matrix.png")

cm_display_1 = metrics.ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(pred_df_all[target_name + '_actual'].astype(str), pred_df_all[target_name].astype(str),normalize = 'true'), display_labels = class_names)
fig_cm_1, ax_cm_1 = plt.subplots()
cm_display_1.plot(ax=ax_cm_1)
mlflow.log_figure(fig_cm_1,"test_confusion_matrix_normalized.png")

# COMMAND ----------

metrics_df = pd.DataFrame([['test_recall',recall], ['test_precision',precision], ['test_f1',f1], ['test_f_beta',fbeta], ['test_auc',auc], ['test_overall_recall',overall_recall],['test_overall_precision',overall_precision],['test_overall_f1',overall_f1],['test_overall_fbeta',overall_fbeta], ['test_accuracy',overall_accuracy]], columns=['Metric', 'Value'])
metrics_df = round(metrics_df,3)
display(metrics_df)

metrics_df.to_csv("metrics_df_test_data.csv",index=False)
mlflow.log_artifact("metrics_df_test_data.csv")

# COMMAND ----------

print("TRAIN")
print(pd.DataFrame(y_train_en)[0].value_counts(dropna=False,normalize=True)*100)

print("TEST DATA ")
print(pred_df_all[target_name +'_actual'].value_counts(dropna=False,normalize=True)*100)
print(pred_df_all[target_name].value_counts(dropna=False,normalize=True)*100)

print("####### VALIDATION #########")
print(pd.DataFrame(y_val_en)[0].value_counts(dropna=False,normalize=True)*100)
print(pd.DataFrame(y_val_pred)[0].value_counts(dropna=False,normalize=True)*100)

# COMMAND ----------

try:
  mlflow.end_run()
except:
  print("exception in ending mlflow")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data describe

# COMMAND ----------

t = list(top_vars.copy())
if 'id' in t:
  t.remove('id')
X_train[t].describe().reset_index()

# COMMAND ----------

X_val[top_vars].describe().reset_index()

# COMMAND ----------

try:
  mlflow.end_run()
  mlflow.start_run(experiment_id=exp_id,run_name='all_data_describe')
  mlflow.autolog()
  mlflow.log_param("best_model",best_model)
except Exception as e:
  print(e)

# COMMAND ----------

t = list(top_vars.copy())
if 'id' in t:
  t.remove('id')
train_data_describe = X_train[t].describe().reset_index()
test_data_describe = pred_df_orig[top_vars].describe().reset_index()
val_data_describe = X_val[top_vars].describe().reset_index()

train_data_describe.to_csv("train_data_describe.csv",index=False)
test_data_describe.to_csv("test_data_describe.csv",index=False)
val_data_describe.to_csv("val_data_describe.csv",index=False)

mlflow.log_artifact("train_data_describe.csv")
mlflow.log_artifact("test_data_describe.csv")
mlflow.log_artifact("val_data_describe.csv")

# COMMAND ----------

print("TRAIN")
print("::actual")
print(pd.DataFrame(y_train_en)[0].value_counts(dropna=False,normalize=True)*100)

print("TEST DATA ")
print("::actual")
print(pred_df_all[target_name+'_actual'].value_counts(dropna=False,normalize=True)*100)
print("::pred")
print(pred_df_all[target_name].value_counts(dropna=False,normalize=True)*100)

print("####### VALIDATION #########")
print("::actual")
print(pd.DataFrame(y_val_en)[0].value_counts(dropna=False,normalize=True)*100)
print("::pred")
print(pd.DataFrame(y_val_pred)[0].value_counts(dropna=False,normalize=True)*100)

# COMMAND ----------

mlflow.log_param("train_actual_target_distr",pd.DataFrame(y_train_en)[0].value_counts(dropna=False,normalize=True).to_dict())

mlflow.log_param("val_actual_target_distr",pd.DataFrame(y_val_en)[0].value_counts(dropna=False,normalize=True).to_dict())
mlflow.log_param("val_pred_target_distr",pd.DataFrame(y_val_pred)[0].value_counts(dropna=False,normalize=True).to_dict())

mlflow.log_param("test_actual_target_distr",pred_df_all[target_name+'_actual'].value_counts(dropna=False,normalize=True).to_dict())
mlflow.log_param("test_pred_target_distr",pred_df_all[target_name].value_counts(dropna=False,normalize=True).to_dict())

# COMMAND ----------

try:
  mlflow.end_run()
except:
  print("exception in ending mlflow")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Persist

# COMMAND ----------

spark.sql("drop table IF EXISTS "+ prediction_table_name+"_gc_results")
pred_df_all_persist = pred_df_all.copy()
pred_df_all_persist['model_id'] = model_ID
pred_df_all_persist.columns = ["pred_"+str(i) if str(i)[0] in ['1','2','3','4','5','6','7','8','9','0'] else str(i) for i in pred_df_all_persist.columns] #make columns starting with a number , preds_number
pred_df_all_persist_spark = spark.createDataFrame(pred_df_all_persist)

if(output_to == "redshift"):
  try:
    persist_data(pred_df_all_persist_spark, prediction_table_name+"_gc_results", ['model_id','rid'],is_rs = True)
  except Exception as e:
    print(e)
    print("PERSIST FAILED !!!!")

if(output_to == "spark"):
  try:
    persist_data(pred_df_all_persist_spark, prediction_table_name+"_gc_results", ['model_id','rid'],is_rs = False)
  except Exception as e:
    print(e)
    print("PERSIST FAILED !!!")

# COMMAND ----------

# DBTITLE 1, Conclusion
displayHTML(" Prediction Results are Persisted in :  "+prediction_table_name)
mlflow.end_run()
logging.shutdown()

# COMMAND ----------


