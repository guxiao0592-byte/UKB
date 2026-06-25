import os
import time
import numpy as np
import pandas as pd
import configparser
from data_utils import *


#### The number of headers to trigger a batch-loading operation
MAX_LOAD_HEADERS = 512
#### Max string length (for any variable)
MAX_STRING_LENGTH = 50

#### Data preparation: saving CSV data to separate numpy files to accelerate data loading
def data_preparation(data_config_filename, raw_data_dir, numpy_data_dir):
    if not os.path.exists(numpy_data_dir):
        os.mkdir(numpy_data_dir)
    start_time = time.time()
    data_config = initialize_config()
    for _, folders, _ in os.walk(raw_data_dir):
        for foldername in sorted(folders):
            if foldername[0] == ".":
                continue
            folder_dir = os.path.join(raw_data_dir, foldername)
            for _, _, files in os.walk(folder_dir):
                for filename in sorted(files):
                    if not filename.endswith(".csv"):
                        continue
                    print("Loading data from file %s." % (filename))
                    csv_file = os.path.join(folder_dir, filename)
                    headers = pd.read_csv(csv_file, nrows=0).columns.tolist()
                    field_id_batch, count_batch, load_headers = [], [], []
                    curr_index = 1
                    while curr_index < len(headers):
                        field_id, offset_ins, offset_arr = get_header_info(headers[curr_index])
                        ins = offset_ins
                        arr = offset_arr
                        headers_ = [headers[curr_index]]
                        while True:
                            curr_index += 1
                            if curr_index == len(headers):
                                break
                            field_id_, ins_, arr_ = get_header_info(headers[curr_index])
                            if field_id_ != field_id:
                                break
                            ins = ins_
                            arr = arr_
                            headers_ = headers_ + [headers[curr_index]]
                        instance_count = ins - offset_ins + 1
                        array_count = arr - offset_arr + 1
                        assert instance_count * array_count == len(headers_)
                        for h in range(len(headers_)):
                            _, ins_, arr_ = get_header_info(headers_[h])
                            assert ins_ - offset_ins == h // array_count and arr_ - offset_arr == h % array_count
                        numpy_data_file = numpy_data_filename(numpy_data_dir, str(field_id))
                        if not os.path.exists(numpy_data_file):
                            field_id_batch = field_id_batch + [str(field_id)]
                            count_batch = count_batch + [len(headers_)]
                            load_headers = load_headers + headers_
                        data_config[field_id] = {
                            VN_FIELDID: str(field_id),
                            VN_INSTANCES: str(instance_count),
                            VN_ARRAY: str(array_count),
                        }
                        if curr_index == len(headers) or len(load_headers) >= MAX_LOAD_HEADERS:
                            if len(load_headers) == 0:
                                continue
                            print("Collected %d columns to load:" % (len(load_headers)), end="")
                            print(load_headers)
                            numpy_data = pd.read_csv(csv_file, usecols=load_headers, dtype=object)
                            assert(numpy_data.shape[0] == N_OVERALL)
                            assert(numpy_data.shape[1] == len(load_headers))
                            curr_index_ = 0
                            for i in range(len(field_id_batch)):
                                numpy_data_file = os.path.join(numpy_data_dir, field_id_batch[i] + ".npz")
                                data = numpy_data[load_headers[curr_index_: curr_index_ + count_batch[i]]].to_numpy(dtype=object, na_value=EMPTY_CELL)
                                curr_index_ += count_batch[i]
                                np.savez_compressed(numpy_data_file, data=data)
                                print("Data loaded for field ID %s." % (field_id_batch[i]))
                            field_id_batch, count_batch, load_headers = [], [], []
    with open(data_config_filename, "w") as configfile:
        data_config.write(configfile)
    print("Data preparation is finished, %f seconds elapsed." % (time.time() - start_time))
    return 0

#### Data validation: confirming that all data (with coding) does not contain unknown values
def data_validation(coding_config_filename, data_config_filename, data_list_filename, numpy_data_dir, simple_mode=False):
    start_time = time.time()
    coding_config = load_config_from_file(coding_config_filename)
    data_config = load_config_from_file(data_config_filename)
    data_list = pd.read_csv(data_list_filename, low_memory=False)
    for l in range(data_list.shape[0]):
        type_ = str(int(data_list[VN_TYPE][l])) if not np.isnan(data_list[VN_TYPE][l]) else EMPTY_CELL
        category = str(int(data_list[VN_CATEGORY][l]))
        field_id = str(int(data_list[VN_FIELDID][l]))
        field = str(data_list[VN_FIELD][l])
        value_type = str(data_list[VN_VALUETYPE][l])
        units = str(data_list[VN_UNITS][l])
        instances = int(data_list[VN_INSTANCES][l])
        array = int(data_list[VN_ARRAY][l])
        coding = NO_CODING_IDENTIFIER if np.isnan(data_list[VN_CODING][l]) else str(int(data_list[VN_CODING][l]))
        if coding != NO_CODING_IDENTIFIER:
            assert coding in coding_config
        if not value_type in SUPPORTED_VT_SET:
            continue
        print("Validating data for field ID %s." % (field_id))
        numpy_data_file = numpy_data_filename(numpy_data_dir, field_id)
        if not os.path.exists(numpy_data_file):
            print("  Warning: data does not exist; skipped.")
            continue
        assert field_id in data_config
        if not simple_mode:
            data = load_numpy_data(numpy_data_file)
            assert data.shape[0] == N_OVERALL
            assert data.shape[1] == int(data_config[field_id][VN_INSTANCES]) * int(data_config[field_id][VN_ARRAY])
            if data.shape[1] != instances * array:
                print("  Warning: data size does not match: %d != %d * %d." % (data.shape[1], instances, array))
            elif instances != int(data_config[field_id][VN_INSTANCES]) or array != int(data_config[field_id][VN_ARRAY]):
                print("  Warning: instances/array does not match: %d != %d or %d != %d." % \
                    (instances, int(data_config[field_id][VN_INSTANCES]), array, int(data_config[field_id][VN_ARRAY])))
        instances = int(data_config[field_id][VN_INSTANCES])
        array = int(data_config[field_id][VN_ARRAY])
        data_config[field_id] = {
            VN_TYPE: type_,
            VN_CATEGORY: category,
            VN_FIELDID: field_id,
            VN_FIELD: data_list[VN_FIELD][l],
            VN_VALUETYPE: value_type,
            VN_UNITS: VN_UNIT_PERCENT if units == VN_UNIT_PERCENT_SYMBOL else units,
            VN_CODING: coding,
            VN_INSTANCES: str(instances),
            VN_ARRAY: str(array),
            VN_FLAG: VN_FLAG_VALUE,
        }
        if simple_mode:
            continue
        sv_dict = coding_config[coding]
        if value_type == VT_CONT or value_type == VT_INTE:
            # This is to confirm that all special values are smaller than normal values.
            data_processed = np.zeros_like(data, dtype=np.float64)
            for ind, _ in np.ndenumerate(data):
                data_processed[ind] = float(data[ind]) if data[ind] != EMPTY_CELL else np.nan
            unique = np.unique(data_processed[~np.isnan(data_processed)])
            sv_dict_float = {float(key): value for key, value in sv_dict.items()}
            is_sv = np.zeros(len(unique), dtype=np.int8)
            for v in range(len(unique)):
                is_sv[v] = 1 if unique[v] in sv_dict_float else 0
            index_left = 0
            while index_left < len(unique) and is_sv[index_left]:
                index_left += 1
            index_right = len(unique) - 1
            while index_right >= 0 and is_sv[index_right]:
                index_right -= 1
            if index_left + (len(unique)-1-index_right) < len(unique):
                for v in range(index_left, index_right+1):
                    if is_sv[v]:
                        print("  Warning: special value %f overflow (numeric) with coding %s." % (unique[v], coding))
                        break
        elif value_type == VT_DATE or value_type == VT_TIME:
            # This is to confirm that all normal values lie between the special values.
            data_processed = np.zeros_like(data, dtype="S"+str(MAX_STRING_LENGTH))
            for ind, _ in np.ndenumerate(data):
                data_processed[ind] = str(data[ind]) if data[ind] != EMPTY_CELL else EMPTY_CELL
            unique = np.unique(data_processed)
            if string_from_bytes(unique[0]) == EMPTY_CELL:
                unique = np.delete(unique, 0)
            is_sv = np.zeros(len(unique), dtype=np.int8)
            for v in range(len(unique)):
                is_sv[v] = 1 if string_from_bytes(unique[v]) in sv_dict else 0
            index_left = 0
            while index_left < len(unique) and is_sv[index_left]:
                index_left += 1
            index_right = len(unique) - 1
            while index_right >= 0 and is_sv[index_right]:
                index_right -= 1
            if index_left + (len(unique)-1-index_right) < len(unique):
                for v in range(index_left, index_right+1):
                    u = string_from_bytes(unique[v])
                    if is_sv[v]:
                        print("  Warning: special value %d overflow (date/time) with coding %s." % (u, coding))
                        break
        elif value_type == VT_CATS or value_type == VT_CATM:
            data_processed = np.zeros_like(data, dtype="S"+str(MAX_STRING_LENGTH))
            for ind, _ in np.ndenumerate(data):
                data_processed[ind] = str(data[ind]) if data[ind] != EMPTY_CELL else EMPTY_CELL
            unique = np.unique(data_processed)
            if string_from_bytes(unique[0]) == EMPTY_CELL:
                unique = np.delete(unique, 0)
            for u in unique:
                u_ = config_parser_key(string_from_bytes(u))
                if not u_ in sv_dict:
                    print("  Warning: undefined value %s (categorical) with coding %s." % (u_, coding))
                    break
        else:
            print("  Exception: undefined feature value type %s, skipped." % (value_type))
            continue
    with open(data_config_filename, "w") as configfile:
        data_config.write(configfile)
    print("Data validation is finished, %f seconds elapsed." % (time.time() - start_time))
    return 0

def feature_extraction_one(coding_config, data_config_section, numpy_data_dir, feature_dir):
    type_ = str(data_config_section[VN_TYPE])
    category = str(data_config_section[VN_CATEGORY])
    field_id = str(data_config_section[VN_FIELDID])
    field = str(data_config_section[VN_FIELD])
    value_type = str(data_config_section[VN_VALUETYPE])
    units = str(data_config_section[VN_UNITS])
    instances = int(data_config_section[VN_INSTANCES])
    array = int(data_config_section[VN_ARRAY])
    coding = str(data_config_section[VN_CODING])
    if field_id in BYPASSING_FIELDID_SET:
        return -2
    if coding != NO_CODING_IDENTIFIER:
        assert coding in coding_config
        if coding in BYPASSING_CODING_SET:
            return -2
    else:
        assert value_type != VT_CATS and value_type != VT_CATM
    if not value_type in SUPPORTED_VT_SET:
        return -2
    feature_file = feature_filename(feature_dir, field_id)
    if not os.path.exists(feature_file):
        print("Extracting features for field ID %s." % (field_id))
        numpy_data_file = numpy_data_filename(numpy_data_dir, field_id)
        if not os.path.exists(numpy_data_file):
            print("  Warning: data does not exist; skipped.")
            return -1
        data = load_numpy_data(numpy_data_file)
        assert data.shape[0] == N_OVERALL
        if field_id =='4079' or field_id == '4080':
            assert data.shape[1] == instances * array * 2 
        else:
            assert data.shape[1] == instances * array
        N = N_OVERALL
        M = instances * array # Only the first instance is used (this can change in the future)
        sv_dict = {} if coding == NO_CODING_IDENTIFIER else coding_config[coding]
        if value_type in {VT_CONT, VT_INTE}:
            feature = np.zeros((N, M*2), dtype=np.float64)
            sv_dict_float = {float(key): value for key, value in sv_dict.items()}
            if has_sv_operator(coding, sv_dict_float, SV_NUMERIC_SD1, SV_NUMERIC_OP, SV_NUMERIC_SPECIAL):
                age_of_visit = get_age_of_visit(feature_dir)
            if has_sv_operator(coding, sv_dict_float, SV_NUMERIC_SD2, SV_NUMERIC_OP, SV_NUMERIC_SPECIAL):
                year_of_visit = get_year_of_visit(feature_dir)
            for n in range(N):
                for m in range(M):
                    if data[n, m] == EMPTY_CELL:
                        feature[n, m*2+1] = 1
                    else:
                        data_value = float(data[n, m])
                        if data_value in sv_dict_float:
                            op = get_sv_operator(coding, data_value, SV_NUMERIC_OP, SV_NUMERIC_SPECIAL)
                            if op in {SV_NUMERIC_NC, SV_NUMERIC_DNK, SV_NUMERIC_DNP, SV_NUMERIC_DNA, \
                                      SV_NUMERIC_DNR, SV_NUMERIC_IC, SV_NUMERIC_NS, SV_NUMERIC_NA}:
                                feature[n, m*2+1] = 1
                            elif op in {SV_NUMERIC_LTO}:
                                feature[n, m*2] = 0.5
                            elif op in {SV_NUMERIC_NO}:
                                feature[n, m*2] = 0
                            elif op in {SV_NUMERIC_TM}:
                                feature[n, m*2] = MAX_NUMERIC_VALUE
                            elif op in {SV_NUMERIC_SD1}:
                                feature[n, m*2] = age_of_visit[n]
                            elif op in {SV_NUMERIC_SD2}:
                                feature[n, m*2] = year_of_visit[n]
                            else:
                                print("  Exception: unrecognized special value operator: %s." % (op))
                                return -1
                        else:
                            feature[n, m*2] = data_value
            if has_sv_operator(coding, sv_dict_float, SV_NUMERIC_TM, SV_NUMERIC_OP, SV_NUMERIC_SPECIAL):
                feature_ = feature[:, 0:2:M*2]
                max_value = np.max(feature_[feature_ != MAX_NUMERIC_VALUE])
                feature[feature == MAX_NUMERIC_VALUE] = max_value + 1
        elif value_type in {VT_DATE, VT_TIME}:
            feature = np.zeros((N, M*2), dtype=np.float64)
            for n in range(N):
                for m in range(M):
                    if data[n, m] == EMPTY_CELL:
                        feature[n, m*2+1] = 1
                    else:
                        data_value = str(data[n, m])
                        if data_value in sv_dict:
                            op = get_sv_operator(coding, data_value, SV_DATETIME_OP, SV_DATETIME_SPECIAL)
                            if op in {SV_DATETIME_DNK, SV_DATETIME_BDB, SV_DATETIME_MDB, SV_DATETIME_YDB, \
                                      SV_DATETIME_BYB, SV_DATETIME_PHF, SV_DATETIME_DNA, SV_DATETIME_NA}:
                                feature[n, m*2+1] = 1
                            else:
                                print("  Exception: unrecognized special value operator: %s." % (op))
                                return -1
                        else:
                            feature[n, m*2] = date_from_string(data_value) if value_type == VT_DATE else time_from_string(data_value)
        elif value_type in {VT_CATS}:
            #### Logic for "categorical multiple" coding:
            # 1. Each regular class label is assigned an ID.
            # 2. The "non-of-the-above", "not-performed", and "unknown" groups are assigned an ID if necessary.
            # 3. No special operations are defined, but can be added in the future.
            v_count = 0
            v_dict = {}
            # First pass: assign ID for regular categorical values (each class occupies an ID)
            for key, _ in sv_dict.items():
                op = get_sv_operator(coding, key, SV_CATS_OP, SV_CATS_SPECIAL)
                if op == SV_NOT_RECOGNIZED:
                    v_dict.update({key: v_count})
                    v_count += 1
            # Second pass: assign ID for special categorical values (each set shares an ID)
            marked = False
            for key, _ in sv_dict.items():
                op = get_sv_operator(coding, key, SV_CATS_OP, SV_CATS_SPECIAL)
                if op in {SV_CATS_NTA}:
                    v_dict.update({key: v_count})
                    marked = True
            v_count = v_count + 1 if marked else v_count
            marked = False
            for key, _ in sv_dict.items():
                op = get_sv_operator(coding, key, SV_CATS_OP, SV_CATS_SPECIAL)
                if op in {SV_CATS_DNP, SV_CATS_NA}:
                    v_dict.update({key: v_count})
                    marked = True
            v_count = v_count + 1 if marked else v_count
            marked = True
            v_dict.update({SV_CATS_DNK: v_count})
            for key, _ in sv_dict.items():
                op = get_sv_operator(coding, key, SV_CATS_OP, SV_CATS_SPECIAL)
                if op in {SV_CATS_DNK, SV_CATS_DNA, SV_CATS_NS, SV_CATS_VAR}:
                    v_dict.update({key: v_count})
            v_count = v_count + 1
            feature = np.zeros((N, M*v_count), dtype=np.float64)
            for n in range(N):
                for m in range(M):
                    if data[n, m] == EMPTY_CELL:
                        feature[n, m*v_count+v_dict[SV_CATS_DNK]] = 1
                    else:
                        data_value = config_parser_key(str(data[n, m]))
                        if data_value in v_dict:
                            feature[n, m*v_count+v_dict[data_value]] = 1
                        else:
                            op = get_sv_operator(coding, data_value, SV_CATS_OP, SV_CATS_SPECIAL)
                            if op in {}: # A dummy section, left for future extensions
                                ...
                            else:
                                print("  Exception: unrecognized special value operator: %s." % (op))
                                return -1
        elif value_type in {VT_CATM}:
            #### Logic for "categorical multiple" coding:
            # 1. A sample is initialized as "all elements are unknown".
            # 2. A special value defines the elements in the group it corresponds to.
            # 3. A regular value sets the corresponding element as "positive".
            # 4. If some regular values exist in a group, other elements are set as "negative".
            v_count = 0
            v_dict = {}
            # First pass: assign ID for regular categorical values (each class occupies an ID)
            for key, _ in sv_dict.items():
                op = get_sv_operator(coding, key, SV_CATM_OP, SV_CATM_SPECIAL)
                if op == SV_NOT_RECOGNIZED:
                    v_dict.update({key: v_count})
                    v_count += 1
            # Second pass: assign ID for special categorical values (each set shares an ID)
            marked = False
            for key, _ in sv_dict.items():
                op = get_sv_operator(coding, key, SV_CATM_OP, SV_CATM_SPECIAL)
                if op in {SV_CATM_DNP, SV_CATM_NA, SV_CATM_UNC}:
                    v_dict.update({key: v_count})
                    marked = True
            v_count = v_count + 1 if marked else v_count
            feature = np.zeros((N, v_count*2*instances), dtype=np.float64)
            for n in range(N):
                for m in range(M):
                    k =m//array
                    if data[n, m] == EMPTY_CELL:
                        continue
                    else:
                        data_value = str(data[n, m])
                        if data_value in v_dict:
                            feature[n, v_dict[data_value]*2+v_count*k*2] = 1
                        else:
                            op = get_sv_operator(coding, data_value, SV_CATM_OP, SV_CATM_SPECIAL)
                            assert not op in {SV_CATM_DNP, SV_CATM_NA, SV_CATM_UNC}
                            if op in {SV_CATM_DNK, SV_CATM_DNA}:
                                if coding in SV_CATM_GROUP:
                                    group = SV_CATM_GROUP[coding][data_value]
                                    for key, value in v_dict.items():
                                        if key in group:
                                            feature[n, value*2+1+v_count*k*2] = 1
                                else:
                                    for v in range(v_count):
                                        feature[n, v*2+1+v_count*k*2] = 1
                            elif op in {SV_CATM_NTA}:
                                if coding in SV_CATM_GROUP:
                                    group = SV_CATM_GROUP[coding][data_value]
                                    for key, value in v_dict.items():
                                        if key in group:
                                            feature[n, value*2+v_count*k*2] = 0
                                else:
                                    for v in range(v_count):
                                        feature[n, v*2+v_count*k*2] = 0
                            elif op in {SV_CATM_ATA}:
                                if coding in SV_CATM_GROUP:
                                    group = SV_CATM_GROUP[coding][data_value]
                                    for key, value in v_dict.items():
                                        if key in group:
                                            feature[n, value*2+v_count*k*2] = 1
                                else:
                                    for v in range(v_count):
                                        feature[n, v*2+v_count*k*2] = 1
                            else:
                                print("  Exception: unrecognized special value operator: %s." % (op))
                                return -1
        else:
            print("  Exception: undefined feature value type %s, skipped." % (value_type))
            return -1
        np.savez_compressed(feature_file, feature=feature)


def feature_extraction_one_bk(coding_config, data_config_section, numpy_data_dir, feature_dir):
    type_ = str(data_config_section[VN_TYPE])
    category = str(data_config_section[VN_CATEGORY])
    field_id = str(data_config_section[VN_FIELDID])
    field = str(data_config_section[VN_FIELD])
    value_type = str(data_config_section[VN_VALUETYPE])
    units = str(data_config_section[VN_UNITS])
    instances = int(data_config_section[VN_INSTANCES])
    array = int(data_config_section[VN_ARRAY])
    coding = str(data_config_section[VN_CODING])
    if field_id in BYPASSING_FIELDID_SET:
        return -2
    if coding != NO_CODING_IDENTIFIER:
        assert coding in coding_config
        if coding in BYPASSING_CODING_SET:
            return -2
    else:
        assert value_type != VT_CATS and value_type != VT_CATM
    if not value_type in SUPPORTED_VT_SET:
        return -2
    feature_file = feature_filename(feature_dir, field_id)
    if not os.path.exists(feature_file):
        print("Extracting features for field ID %s." % (field_id))
        numpy_data_file = numpy_data_filename(numpy_data_dir, field_id)
        if not os.path.exists(numpy_data_file):
            print("  Warning: data does not exist; skipped.")
            return -1
        data = load_numpy_data(numpy_data_file)
        assert data.shape[0] == N_OVERALL
        if field_id =='4079' or field_id == '4080':
            assert data.shape[1] == instances * array * 2 
        else:
            assert data.shape[1] == instances * array
        N = N_OVERALL
        M = instances * array # Only the first instance is used (this can change in the future)
        sv_dict = {} if coding == NO_CODING_IDENTIFIER else coding_config[coding]
        if value_type in {VT_CONT, VT_INTE}:
            feature = np.zeros((N, M*2), dtype=np.float64)
            sv_dict_float = {float(key): value for key, value in sv_dict.items()}
            if has_sv_operator(coding, sv_dict_float, SV_NUMERIC_SD1, SV_NUMERIC_OP, SV_NUMERIC_SPECIAL):
                age_of_visit = get_age_of_visit(feature_dir)
            if has_sv_operator(coding, sv_dict_float, SV_NUMERIC_SD2, SV_NUMERIC_OP, SV_NUMERIC_SPECIAL):
                year_of_visit = get_year_of_visit(feature_dir)
            for n in range(N):
                for m in range(M):
                    if data[n, m] == EMPTY_CELL:
                        feature[n, m*2+1] = 1
                    else:
                        data_value = float(data[n, m])
                        if data_value in sv_dict_float:
                            op = get_sv_operator(coding, data_value, SV_NUMERIC_OP, SV_NUMERIC_SPECIAL)
                            if op in {SV_NUMERIC_NC, SV_NUMERIC_DNK, SV_NUMERIC_DNP, SV_NUMERIC_DNA, \
                                      SV_NUMERIC_DNR, SV_NUMERIC_IC, SV_NUMERIC_NS, SV_NUMERIC_NA}:
                                feature[n, m*2+1] = 1
                            elif op in {SV_NUMERIC_LTO}:
                                feature[n, m*2] = 0.5
                            elif op in {SV_NUMERIC_NO}:
                                feature[n, m*2] = 0
                            elif op in {SV_NUMERIC_TM}:
                                feature[n, m*2] = MAX_NUMERIC_VALUE
                            elif op in {SV_NUMERIC_SD1}:
                                feature[n, m*2] = age_of_visit[n]
                            elif op in {SV_NUMERIC_SD2}:
                                feature[n, m*2] = year_of_visit[n]
                            else:
                                print("  Exception: unrecognized special value operator: %s." % (op))
                                return -1
                        else:
                            feature[n, m*2] = data_value
            if has_sv_operator(coding, sv_dict_float, SV_NUMERIC_TM, SV_NUMERIC_OP, SV_NUMERIC_SPECIAL):
                feature_ = feature[:, 0:2:M*2]
                max_value = np.max(feature_[feature_ != MAX_NUMERIC_VALUE])
                feature[feature == MAX_NUMERIC_VALUE] = max_value + 1
        elif value_type in {VT_DATE, VT_TIME}:
            feature = np.zeros((N, M*2), dtype=np.float64)
            for n in range(N):
                for m in range(M):
                    if data[n, m] == EMPTY_CELL:
                        feature[n, m*2+1] = 1
                    else:
                        data_value = str(data[n, m])
                        if data_value in sv_dict:
                            op = get_sv_operator(coding, data_value, SV_DATETIME_OP, SV_DATETIME_SPECIAL)
                            if op in {SV_DATETIME_DNK, SV_DATETIME_BDB, SV_DATETIME_MDB, SV_DATETIME_YDB, \
                                      SV_DATETIME_BYB, SV_DATETIME_PHF, SV_DATETIME_DNA, SV_DATETIME_NA}:
                                feature[n, m*2+1] = 1
                            else:
                                print("  Exception: unrecognized special value operator: %s." % (op))
                                return -1
                        else:
                            feature[n, m*2] = date_from_string(data_value) if value_type == VT_DATE else time_from_string(data_value)
        elif value_type in {VT_CATS}:
            #### Logic for "categorical multiple" coding:
            # 1. Each regular class label is assigned an ID.
            # 2. The "non-of-the-above", "not-performed", and "unknown" groups are assigned an ID if necessary.
            # 3. No special operations are defined, but can be added in the future.
            v_count = 0
            v_dict = {}
            # First pass: assign ID for regular categorical values (each class occupies an ID)
            for key, _ in sv_dict.items():
                op = get_sv_operator(coding, key, SV_CATS_OP, SV_CATS_SPECIAL)
                if op == SV_NOT_RECOGNIZED:
                    v_dict.update({key: v_count})
                    v_count += 1
            # Second pass: assign ID for special categorical values (each set shares an ID)
            marked = False
            for key, _ in sv_dict.items():
                op = get_sv_operator(coding, key, SV_CATS_OP, SV_CATS_SPECIAL)
                if op in {SV_CATS_NTA}:
                    v_dict.update({key: v_count})
                    marked = True
            v_count = v_count + 1 if marked else v_count
            marked = False
            for key, _ in sv_dict.items():
                op = get_sv_operator(coding, key, SV_CATS_OP, SV_CATS_SPECIAL)
                if op in {SV_CATS_DNP, SV_CATS_NA}:
                    v_dict.update({key: v_count})
                    marked = True
            v_count = v_count + 1 if marked else v_count
            marked = True
            v_dict.update({SV_CATS_DNK: v_count})
            for key, _ in sv_dict.items():
                op = get_sv_operator(coding, key, SV_CATS_OP, SV_CATS_SPECIAL)
                if op in {SV_CATS_DNK, SV_CATS_DNA, SV_CATS_NS, SV_CATS_VAR}:
                    v_dict.update({key: v_count})
            v_count = v_count + 1
            feature = np.zeros((N, M*v_count), dtype=np.float64)
            for n in range(N):
                for m in range(M):
                    if data[n, m] == EMPTY_CELL:
                        feature[n, m*v_count+v_dict[SV_CATS_DNK]] = 1
                    else:
                        data_value = config_parser_key(str(data[n, m]))
                        if data_value in v_dict:
                            feature[n, m*v_count+v_dict[data_value]] = 1
                        else:
                            op = get_sv_operator(coding, data_value, SV_CATS_OP, SV_CATS_SPECIAL)
                            if op in {}: # A dummy section, left for future extensions
                                ...
                            else:
                                print("  Exception: unrecognized special value operator: %s." % (op))
                                return -1
        elif value_type in {VT_CATM}:
            #### Logic for "categorical multiple" coding:
            # 1. A sample is initialized as "all elements are unknown".
            # 2. A special value defines the elements in the group it corresponds to.
            # 3. A regular value sets the corresponding element as "positive".
            # 4. If some regular values exist in a group, other elements are set as "negative".
            v_count = 0
            v_dict = {}
            # First pass: assign ID for regular categorical values (each class occupies an ID)
            for key, _ in sv_dict.items():
                op = get_sv_operator(coding, key, SV_CATM_OP, SV_CATM_SPECIAL)
                if op == SV_NOT_RECOGNIZED:
                    v_dict.update({key: v_count})
                    v_count += 1
            # Second pass: assign ID for special categorical values (each set shares an ID)
            marked = False
            for key, _ in sv_dict.items():
                op = get_sv_operator(coding, key, SV_CATM_OP, SV_CATM_SPECIAL)
                if op in {SV_CATM_DNP, SV_CATM_NA, SV_CATM_UNC}:
                    v_dict.update({key: v_count})
                    marked = True
            v_count = v_count + 1 if marked else v_count
            feature = np.zeros((N, v_count*2), dtype=np.float64)
            for n in range(N):
                for m in range(M):
                    if data[n, m] == EMPTY_CELL:
                        break
                    else:
                        data_value = str(data[n, m])
                        if data_value in v_dict:
                            feature[n, v_dict[data_value]*2] = 1
                        else:
                            op = get_sv_operator(coding, data_value, SV_CATM_OP, SV_CATM_SPECIAL)
                            assert not op in {SV_CATM_DNP, SV_CATM_NA, SV_CATM_UNC}
                            if op in {SV_CATM_DNK, SV_CATM_DNA}:
                                if coding in SV_CATM_GROUP:
                                    group = SV_CATM_GROUP[coding][data_value]
                                    for key, value in v_dict.items():
                                        if key in group:
                                            feature[n, value*2+1] = 1
                                else:
                                    for v in range(v_count):
                                        feature[n, v*2+1] = 1
                            elif op in {SV_CATM_NTA}:
                                if coding in SV_CATM_GROUP:
                                    group = SV_CATM_GROUP[coding][data_value]
                                    for key, value in v_dict.items():
                                        if key in group:
                                            feature[n, value*2] = 0
                                else:
                                    for v in range(v_count):
                                        feature[n, v*2] = 0
                            elif op in {SV_CATM_ATA}:
                                if coding in SV_CATM_GROUP:
                                    group = SV_CATM_GROUP[coding][data_value]
                                    for key, value in v_dict.items():
                                        if key in group:
                                            feature[n, value*2] = 1
                                else:
                                    for v in range(v_count):
                                        feature[n, v*2] = 1
                            else:
                                print("  Exception: unrecognized special value operator: %s." % (op))
                                return -1
        else:
            print("  Exception: undefined feature value type %s, skipped." % (value_type))
            return -1
        np.savez_compressed(feature_file, feature=feature)

def feature_extraction(coding_config_filename, data_config_filename, numpy_data_dir, feature_dir):
    if not os.path.exists(feature_dir):
        os.mkdir(feature_dir)
    start_time = time.time()
    coding_config = load_config_from_file(coding_config_filename)
    data_config = load_config_from_file(data_config_filename)
    # Extracting these features first for calculating age and/or year (when necessary)
    feature_extraction_one(coding_config, data_config[FIELD_ID_YOB], numpy_data_dir, feature_dir)
    data_config[FIELD_ID_MOB].update({VN_VALUETYPE: VT_INTE}) # Casting the value type of MOB to be integer
    data_config[FIELD_ID_MOB].update({VN_CODING: NO_CODING_IDENTIFIER}) # Removing the coding identifier
    feature_extraction_one(coding_config, data_config[FIELD_ID_MOB], numpy_data_dir, feature_dir)
    feature_extraction_one(coding_config, data_config[FIELD_ID_DOV], numpy_data_dir, feature_dir)
    # The regular procedure for extracting all features
    for s in data_config.sections():
        if VN_FLAG in data_config[s]:
            feature_extraction_one(coding_config, data_config[s], numpy_data_dir, feature_dir)
    print("Feature extraction is finished, %f seconds elapsed." % (time.time() - start_time))
    return 0
