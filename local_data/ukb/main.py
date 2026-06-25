import os
import math
import time
import datetime
import numpy as np
import pandas as pd

from data_utils import *
from info_parser import *
from data_prep import *
from prediction import *

if __name__ == "__main__":
    #coding_rule_parser(CODING_CONFIG_FILENAME, RAW_DATA_DIR, DATA_LIST_FILENAME, CODING_DATA_DIR)
    #data_preparation(DATA_CONFIG_FILENAME, RAW_DATA_DIR, NUMPY_DATA_DIR)
    #data_validation(CODING_CONFIG_FILENAME, DATA_CONFIG_FILENAME, DATA_LIST_FILENAME, NUMPY_DATA_DIR, simple_mode=True)
    #feature_extraction(CODING_CONFIG_FILENAME, DATA_CONFIG_FILENAME, NUMPY_DATA_DIR, ALL_FEATURE_DIR)
    ####################################################################################################
    #selected_field_ids1 = {"21022", "400", "20002", "404", "6671", "20110", "30710", "137", "30650", "30620"}
    # selected_field_ids1 = {"1807","6150"}
    #selected_field_ids1 = ["21022",'3526','1807','20161','1200','2178','2986','1835','2897','3581','1160','6150','6138','2405','20162','3456'] # 2897
        #selected_field_ids1 = {"21022",'3526','1807','20161','1200','2178','2986','1835','2897','3581','6138'} # 2897
    #selected_field_ids2 = {"31", "23111", "137", "404", "3064", "3526", "2188", "30040", "40007", "30720"}    
    ####################################################################################################
    #ensemble_classification(DATA_CONFIG_FILENAME, NUMPY_DATA_DIR, FEATURE_DIR, {VN_TYPE: {TYPE_PHASE1_FEATURE}}, FIELD_ID_ALLCAUSE, MAX_AGE, \
                            #using_unlabeled=False, filter_set=[], max_feature_dims=[4096, 0], max_batch=1)
    #ensemble_classification(DATA_CONFIG_FILENAME, NUMPY_DATA_DIR, FEATURE_DIR, {VN_TYPE: {TYPE_PHASE1_FEATURE}}, FIELD_ID_ALLCAUSE, MAX_AGE, \
                           # using_unlabeled=False, filter_set=[], max_feature_dims=[3072, 0], max_batch=1)

    #ensemble_classification(DATA_CONFIG_FILENAME, NUMPY_DATA_DIR, FEATURE_DIR, {VN_TYPE: {TYPE_PHASE1_FEATURE}, VN_CATEGORY: CATEGORY_SET_MR}, FIELD_ID_ALLCAUSE, MAX_AGE, \
                            #using_unlabeled=False, filter_set=["21022", "400", "20002", "404", "6671", "20110", "30710", "137", "30650", "30620"], max_feature_dims=[3072, 0], max_batch=1)
    ####################################################################################################
    #ensemble_age_estimation(DATA_CONFIG_FILENAME, NUMPY_DATA_DIR, FEATURE_DIR, {VN_CATEGORY: CATEGORY_SET_HOME}, FIELD_ID_ALLCAUSE, [55, 60, 65, 70, 75, 80, 85, 90], \
    #                        filter_set=[], max_feature_dims=[3072, 0], max_batch=1, classifier_list=[CLS_LGBM], max_iterations=1)
                                
    #ensemble_classification(DATA_CONFIG_FILENAME, NUMPY_DATA_DIR, ALL_FEATURE_DIR, {VN_FIELDID: selected_field_ids1}, FIELD_ID_ALLCAUSE, MAX_AGE, \
                           #using_unlabeled=False, filter_set=[], classifier_list=[CLS_LGBM],max_feature_dims=[3072, 0], max_batch=1)
    """
    selected_field_ids1 = get_field("F:/CHARLS_result/Field.wave4.new.sort.txt")

    X_vdt_data = pd.read_csv("F:/CHARLS_result/x.wave4.test.csv",sep="\t")
    y_df = pd.read_csv("F:/CHARLS_result/y.wave4.test.csv",sep="\t")
    
    missing_fraction = X_vdt_data.isnull().mean(axis=1)
    valid_rows = missing_fraction <= 1
    X_vdt_data = X_vdt_data[valid_rows]
    X_vdt_data = X_vdt_data.reset_index(drop=True)
 
    y_df= y_df[valid_rows]
    y_df = y_df.reset_index(drop=True)
    y_df.columns = [re.split('_', col)[-1] for col in y_df.columns]
    #print(missing_fraction)
    
    #X_vdt =X_df.to_numpy()
    y_vdt =y_df.to_numpy()
    list =[]
    """
    #selected_field_ids1 = {"21022"}
    X_vdt_data =[]
    y_vdt =[]
    #selected_field_ids1 = ["21022",'3526','1807','20161','1200','2178','2986','1835','2897','3581','1160','6150','6138','2405','20162','3456']
    selected_field_ids1 = {"21022","1807"}
    ukb_hox_test(DATA_CONFIG_FILENAME, NUMPY_DATA_DIR, ALL_FEATURE_DIR, {VN_FIELDID: selected_field_ids1}, FIELD_ID_ALLCAUSE, MAX_AGE, X_vdt_data,y_vdt, \
               using_unlabeled=False, filter_set=[], classifier_list=[CLS_LGBM],max_feature_dims=[3072, 0], max_batch=1)
    
    #ukb_format(DATA_CONFIG_FILENAME, NUMPY_DATA_DIR, ALL_FEATURE_DIR, {VN_TYPE: {TYPE_PHASE1_FEATURE}}, FIELD_ID_ALLCAUSE, MAX_AGE, X_vdt_data,y_vdt, \
               #using_unlabeled=False, filter_set=[], classifier_list=[CLS_LGBM],max_feature_dims=[3072,0], max_batch=2)
    # (s_test_o,s_test_b,s_vdt_o,s_vdt_b) 
    """
    # 梯度测试
    with open("F:/CHARLS_result/wave3.2015.score.csv",'w') as f :
        f.write("num\tscore_test_original\tscore_test_score_test_balanced\tscore_vdt_original\tscore_vdt_balanced\tauc_vdt_original\tauc_vdt_balanced\tauc_test_original\tauc_test_balanced\n")
        for i in iter(selected_field_ids1):
            #list = []
            list.append(i)
            num =len(list)
            print(list)
            (s_test_o,s_test_b,s_vdt_o,s_vdt_b,auc_vdt_o,auc_vdt_b,auc_test_o,auc_test_b) = vdt_ensemble_classification(DATA_CONFIG_FILENAME, NUMPY_DATA_DIR, FEATURE_DIR, {VN_FIELDID: list}, FIELD_ID_ALLCAUSE, MAX_AGE, X_vdt_data,y_vdt, \
                            using_unlabeled=False, filter_set=[], classifier_list=[CLS_LGBM],max_feature_dims=[3072, 0], max_batch=1)
            f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(num,s_test_o,s_test_b,s_vdt_o,s_vdt_b,auc_vdt_o,auc_vdt_b,auc_test_o,auc_test_b))
    """
    #(s_test_o,s_test_b,s_vdt_o,s_vdt_b,auc_vdt_o,auc_vdt_b,auc_test_o,auc_test_b) = vdt_ensemble_classification(DATA_CONFIG_FILENAME, NUMPY_DATA_DIR, FEATURE_DIR, {VN_FIELDID: selected_field_ids1}, FIELD_ID_ALLCAUSE, MAX_AGE, X_vdt_data,y_vdt, \
                         #using_unlabeled=False, filter_set=[], classifier_list=[CLS_LGBM],max_feature_dims=[3072, 0], max_batch=1)

    
    #ensemble_age_estimation(DATA_CONFIG_FILENAME, NUMPY_DATA_DIR, FEATURE_DIR, {VN_CATEGORY: CATEGORY_SET_HOME}, FIELD_ID_ALLCAUSE, [55, 60, 65, 70, 75, 80, 85, 90], \
    #                        filter_set=[], max_feature_dims=[3072, 0], max_batch=1, classifier_list=[CLS_LGBM], max_iterations=1)
    #ensemble_age_estimation(DATA_CONFIG_FILENAME, NUMPY_DATA_DIR, FEATURE_DIR, {VN_FIELDID: selected_field_ids1}, FIELD_ID_ALLCAUSE, [45, 50,55, 60, 65, 70, 75, 80, 85], \
                            #filter_set=[], max_feature_dims=[3072, 0], max_batch=1, classifier_list=[CLS_LGBM], max_iterations=1)
    #ensemble_age_estimation(DATA_CONFIG_FILENAME, NUMPY_DATA_DIR, FEATURE_DIR, {VN_TYPE: {TYPE_PHASE1_FEATURE,TYPE_PHASE2_FEATURE,TYPE_CONTROL_VAR,TYPE_RESULT_VAR}}, FIELD_ID_ALLCAUSE, [55, 60, 65, 70, 75, 80, 85, 90], \
                            #filter_set=[], max_feature_dims=[3072, 0], max_batch=1, classifier_list=[CLS_LGBM], max_iterations=1)