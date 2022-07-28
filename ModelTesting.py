# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 15:19:41 2022

@author: nchong
"""
import numpy as np
import pandas as pd
import joblib
import os
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

#--------------------------Functions---------------------------#
def create_speckle_column(df):
    """
    create speckle column from DELTA
    """
    df["SPECKLE"] = np.where(df["DELTA"]==0,0,1)
    cols = df.columns.tolist()
    cols = [cols[-1]]+cols[:-1] #move speckle col to the front
    df = df.reindex(columns=cols)
    print("Distribution of data:", Counter(df["SPECKLE"]))
    
    return df


def featuremapping(df,feature_mapping,old,new):
    """
    Map feature names
    
    df[DataFrame]: data where feature names are to be mapped
    feature_mapping[DataFrame]: feature file that contains the old and new feature names
    old[str]: name of the old feature column
    new[str]: name of the new feature column
    """
    old_col = feature_mapping[old].values.tolist()
    new_col = feature_mapping[new].values.tolist()
  
    df.rename(
        columns={i:j for i,j in zip(new_col,old_col)}, inplace=True
    )
    return df

def check_duplicates(df,output_path):
    """
    Check for duplicates in VID 
    df[DataFrame]: input df    
    """
    if df['VID'].duplicated().any():
        print("There are duplicates in VID")
        df_dup = df[df.duplicated(subset=['VID'])] #get the duplicates in df, returns repeated rows only 
        df= df[df["VID"].isin(df_dup["VID"])] #get all the duplicates in df, returns original + repeated rows 
        df.to_csv(output_path + "duplicates.csv", index=False)
        
    else:
        print("There are no duplicates in VID")
        
def check_colna(df,output_path):
    """
    Check each column for nulls. Returns Feature, total null and % of null for each column
    df[DataFrame]: dataframe
    
    """
    colna_df = pd.DataFrame(columns =["Feature", "Total Null", "% of Null"])
    for col in df.columns: 
        #checking if there is any null in the column

        if df[col].isnull().sum()>0: 
            
            # if null present, total number of null in the column stores here
            total_null = df[col].isnull().sum() 
            new_row = {'Feature':col, 'Total Null':total_null, '% of Null':total_null*100/len(df)}
            #append row to the dataframe
            colna_df = colna_df.append(new_row, ignore_index=True)
            
    colna_df= colna_df.sort_values("% of Null", ascending=False)    
     
    if colna_df.empty:
        print('No NA columns!')
    else:
        print('There are NA columns!')
        colna_df.to_csv(output_path + "NA_Columns.csv", index=False) 

def check_rowna(df,output_path):  
    """
    Check each row for nulls.Returns VID, total null and % of null for each row
    df[DataFrame]: dataframe
    
    """
      
    colrow_df = pd.DataFrame(columns =["SPECKLE","VID", "Total Null", "% of Null"])
    for i in df.index: 
        #checking if there is any null in the row
        if df.iloc[i].isnull().sum()>0:             
            # if null present, total number of null in the row stores here
            total_null = df.iloc[i].isnull().sum() 
            new_row = {'SPECKLE':df.iloc[i,0],'VID':df.iloc[i,1], 'Total Null':total_null, '% of Null':round(total_null*100/(len(df.columns)-2),2)}
            #append row to the dataframe
            colrow_df = colrow_df.append(new_row, ignore_index=True)
            
    colrow_df= colrow_df.sort_values("% of Null", ascending=False)   
    
    if colrow_df.empty:
        print('No more NA rows!')
    else:
        print('There are NA rows!')
        colrow_df.to_csv(output_path + "NA_Rows.csv", index=False)    

def convert_neg_to_pos(df,cols_to_keep):
    """
    convert negative columns to positive - except for IDV and HVQK columns
    df: dataframe
    cols_to_keep[tuple]: tokens for columns which are not converted to negative
    """
    #df with tokens which are not converted to negative
    df_keep = df[df.columns[df.columns.str.startswith(cols_to_keep)]]
    # print("df_keep shape",df_keep.shape)
    
    # Create df_to_convert - drop columns from cols_to_keep
    df_to_convert = df.drop([col for col in df if col.startswith(cols_to_keep)], axis=1)    
    # print("df_to_convert shape",df_to_convert.shape)
    
    #Convert negative columns in df_to_convert to positive
    df_positive = df_to_convert.abs()
    # print("df_positive shape",df_to_convert.shape)
    
    #check whether there's any negative value left in df_positive
    # print("Is there negative value left:",(df_positive < 0).any().any())
    
    #Concatenating df_keep and df_positive along columns
    df = pd.concat([df_keep, df_positive], axis=1)
    print("Shape after converting neg to pos:", df.shape)
    
    return df

def check_col_negative(df,output_path,value):
    """
    Check each column for negative values. Returns Feature, total negative values and % of negative values for each column
    df[DataFrame]: dataframe
    
    """
    col_negative_df = pd.DataFrame(columns =["Feature", "Total Negative Values", "% of Negative Values"])
    for col in df.columns: 
        #checking if there is any specific negative value in the column

        if df[col].isin([value]).sum()>0: 
            
            # if specific negative value present, total number of specific negative value in the column stores here
            total_negative = df[col].isin([value]).sum() 
            new_row = {'Feature':col, 'Total Negative Values':total_negative, '% of Negative Values':round(total_negative*100/len(df),2)}
            #append row to the dataframe
            col_negative_df = col_negative_df.append(new_row, ignore_index=True)
            
    col_negative_df = col_negative_df.sort_values("% of Negative Values", ascending=False)    
    col_negative_df.to_csv(output_path + "Check%of"+str(value)+"values.csv", index=False) 

def Negative_value_impute(df,value,imptype):
    """
    Impute Negative value (can choose negative value to impute)
    df[DataFrame]:df
    imptype[string]: "mean" to impute data with mean, "median" to impute data with median
    """
    if imptype == "mean":        
        df = df.replace(value,df.mean())
    if imptype == "median":
        df = df.replace(value,df.median())
    if imptype == "zero":
        df = df.replace(value,0)
    return df

def unary(df):
    """
    Checks for unary columns.
    df[DataFrame]: input dataframe
    """
    unarycolumns = [col for col in df.columns if len(df[col].unique())==1]
    if unarycolumns:
        print("The unary column are:",unarycolumns)        
    else:
        print("There are no unary columns!")
        
def scale_data(X_train,X_test):
    """
    Scaling X train and validation with normalization
    params:
    X_train[DataFrame]: input X train
    X_test[DataFrame]: input X validation (test)
    
    """           
    scaler = MinMaxScaler()    
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled,columns= X_train.columns)
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled,columns= X_test.columns)
       
    return X_test_scaled

def prediction(X_test_scaled,X_test_sf,y_test,model,output_path):
    
    modeloutput_path = output_path + "ModelResults/"
    
    if not os.path.exists(modeloutput_path):
        os.makedirs(modeloutput_path)

    # Predicting the classes for validation set
    y_pred = model.predict(X_test_scaled)
    
    print("Distribution of prediction:", Counter(y_pred))
    
    #overall accuracy
    overall_acc = round(metrics.accuracy_score(y_test, y_pred)*100,2)
    overall_acc = {'Overall Acc %':overall_acc}
    overall_acc = pd.DataFrame([overall_acc])
    overall_acc.to_csv(modeloutput_path+"Overall_Accuracy.csv")

    #classification report
    report = metrics.classification_report(y_test, y_pred,zero_division=0,output_dict=True)
    report = pd.DataFrame(report).transpose()
    report.to_csv(modeloutput_path+"Classification_Report.csv")

    #confusion matrix with accuracies for each label
    class_accuracies = []

    for class_ in y_test.sort_values(ascending= True).unique():
        class_acc = round(np.mean(y_pred[y_test == class_] == class_)*100,2)
        class_accuracies.append(class_acc)

    class_acc = pd.DataFrame(class_accuracies,index=['true:0', 'true:1'],columns= ["Accuracy %"])

    cf_matrix = pd.DataFrame(
        metrics.confusion_matrix(y_test, y_pred, labels= [0, 1]), 
        index=['true:0', 'true:1'], 
        columns=['pred:0', 'pred:1']
    )

    ascend = None #input None/True/False to order the confusion matrix
    if ascend == None:
        cf_matrix = pd.concat([cf_matrix,class_acc],axis=1)
    else:
        cf_matrix = pd.concat([cf_matrix,class_acc],axis=1).sort_values(by=['Accuracy %'], ascending=ascend)

    cf_matrix.to_csv(modeloutput_path+"Confusion_Matrix_test.csv")   
    #confusion matrix with accuracies for each label

    #validation results 
    val_results = pd.concat([X_test_sf,X_test_scaled,pd.DataFrame(y_test),pd.DataFrame(y_pred,columns = ["PRED_SPECKLE"])],axis=1)
    val_results.to_csv(modeloutput_path+"Val_results.csv",index=False) 

    print("Model testing completed!")
    
#-------------------------------main------------------------------#

def main(testdata_path,model_path,X_train_path,features_path,output_path,featuremapping_path,old,new):    
    
    #----------read test data-----------# 
    df = pd.read_csv(testdata_path)
    print("Test data read")
    print("Shape of test data:",df.shape)
    
    #---------create output_path if it doesn't exist--#
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    #---------Data Prep starts------#
    #create speckle column from DELTA
    try:
        df = create_speckle_column(df)
    except Exception as e:
        print(e)
        print("Unable to create SPECKLE column")
        return
    
    # map features if needed
    if featuremapping_path != None:
        try:
            feature_mapping = pd.read_csv(featuremapping_path)
            print("Feature mapping file read")
            df = featuremapping(df,feature_mapping,old,new)
            print("Shape of test data after feature mapping:", df.shape)
            
        except Exception as e:
            print(e)
            print("Feature mapping failed!")
            return
            
    #filter the test data with the features used in model building, maintain columns "SPECKLE" and "VID"
    
    features = pd.read_csv(features_path)
    print("Important features file read")
    try:
        df = df[pd.concat([pd.Series("SPECKLE"),pd.Series("VID"),features["Feature"]])]
        print("Shape of test data:",df.shape)
    except Exception as e:
        print(e)
        print("Some feature(s) used for model building are not in test data")
        return
    
    #check for duplicates
    check_duplicates(df,output_path)
    print("Duplicates checked")
    
    #check column for nulls
    check_colna(df,output_path)
    print("Checked columns for NA")
     
    # check rows for nulls
    check_rowna(df,output_path)
    print("Checked rows for NA")
    
    # Create X_test, X_test_sf and target y_test
    y_test = df["SPECKLE"]
    X_test_sf = df["VID"]
    X_test = df.drop(["SPECKLE","VID"],axis=1)
    print("Shape of X_test:", X_test.shape)
    
    # Impute NA with 0
    X_test = X_test.fillna(0)
    check_rowna(X_test,output_path)
    
    #convert negative columns to positive - except for IDV and HVQK columns
    X_test = convert_neg_to_pos(X_test,cols_to_keep=("IDV", "HVQK"))
    print("Negative values converted to positive - except for IDV and HVQK features")
    
    #Checking IDV and HVQK columns for negative values
    check_col_negative(X_test,output_path,value= -5555)
    check_col_negative(X_test,output_path,value= -555,)
    check_col_negative(X_test,output_path,value= -999)
    check_col_negative(X_test,output_path,value = -9999)
    print("Checked IDV and HVQK columns for negative values ")
    
    #Negative value (invalid data) imputation of IDV and HVQK columns 
    X_test = Negative_value_impute(X_test,value=-5555,imptype="zero")
    X_test = Negative_value_impute(X_test,value=-999,imptype="zero")
    X_test = Negative_value_impute(X_test,value=-9999,imptype="zero")    
    X_test = Negative_value_impute(X_test,value=-555,imptype="zero")
    print("Negative value (invalid data) imputation of IDV and HVQK features done")
    print("Is there still negative values in the test data:",(X_test < 0).any().any())
    
    #Normalize the data
    X_train = pd.read_csv(X_train_path)
    print("Train data read!")
    
    try:    
        X_train = X_train[features["Feature"]]
        print("X_train filtered with important features only!")
        print("Shape of X_train:", X_train.shape)
    except Exception as e:
        print(e)
        print("Features in X_train not the same as features used for model building!")
        return
    
    X_test = X_test[X_train.columns] #make sequence of features in test same as train before normalization
    X_test_scaled = scale_data(X_train,X_test)
    print("Shape of X_test_scaled:", X_test_scaled.shape)

    #-------------Model Testing----------#
    model = joblib.load(model_path) #read model to be tested
    prediction(X_test_scaled,X_test_sf,y_test,model,output_path)
   
if __name__ == '__main__': 
    
    """
    testdata_path [str] = path and filename of test data
    model_path [str]= path and filename of model 
    X_train_path [str]= path and filename of train data with negative handling
    features_path[str] = path and filename of features
    output_path[str] = path to write output, folder will be created for output_path if folder doesn't exist yet
    featuremapping_path[str/None] = [str] path and filename of feature mapping file, [None] no feature mapping needed
    old[str/None] = old column name in feature mapping file - should be same feature name as used in model building,[None] no feature mapping needed
    new [str/None] = old column name in feature mapping file - to be mapped to old name, [None] no feature mapping needed
    
    """
    
    path = 'C:/Users/nchong/OneDrive - Intel Corporation/Documents/ML based speckle POC/'
    testdata_path = path + 'DataPreparation/TestData/Set3_Na0/Test_Set3.csv'
    model_path = path + 'ModelBuilding/MergedData_TrainValExchange_Na0/Ensemble300fs/SVM/Weight_6/SVMmodel.joblib'
    X_train_path = path + 'DataPreparation/MergedData_TrainValExchange_Na0/TrainData_NegHandling_WithXY_WithHVQKDiff_Na0.csv'
    features_path = path + 'FeatureSelection/SNR_R5_ww51.4/EnsembleTop300Fs.csv'
    output_path = path + 'DataPreparation/C0_Testing/'
    featuremapping_path= path + 'DataPreparation/TestData/Set3_Na0/FeatureMapping_C0_B3_Edited.csv'
    old = 'B3'
    new = 'C0'
    # featuremapping_path= None
    # old = None
    # new = None
    
    main(testdata_path = testdata_path,model_path = model_path,X_train_path=X_train_path,features_path=features_path,output_path=output_path,featuremapping_path=featuremapping_path,old=old,new=new)
     
 