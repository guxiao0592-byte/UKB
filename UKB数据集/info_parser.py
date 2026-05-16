import os
import numpy as np
import pandas as pd
import urllib.request
import configparser

from data_utils import *


CODING_DOWNLOAD_URL_PREFIX = "https://biobank.ndph.ox.ac.uk/ukb/coding.cgi?id="
CODING_DOWNLOAD_URL_SUFFIX = "&nl=1"
TAG_TABLE_START = "<table"
TAG_TABLE_END = "</table>"
HEADER_CODING = "Coding"
HEADER_MEANING = "Meaning"
HEADER_SELECTABLE = "Selectable"
TEXT_SELECTABLE_NO = "No"
SPECIAL_DESCRIPTION_DUPLICATED = "DUPLICATED!!!"

def get_coding_filename(coding_data_dir, coding):
    return os.path.join(coding_data_dir, coding + ".txt")

def coding_texts_download(coding_config_filename, raw_data_dir, data_list_filename, coding_data_dir):
    if not os.path.exists(coding_data_dir):
        os.mkdir(coding_data_dir)
    data_list = pd.read_csv(data_list_filename, low_memory=False)
    coding_list = np.zeros((data_list.shape[0]), dtype=np.int32)
    count = 0
    for l in range(data_list.shape[0]):
        if not np.isnan(data_list[VN_CODING][l]):
            coding_list[count] = int(data_list[VN_CODING][l])
            count += 1
    unique = np.unique(coding_list[0: count])
    coding_config = initialize_config()
    for coding in unique:
        coding_filename = get_coding_filename(coding_data_dir, str(coding))
        if not os.path.exists(coding_filename):
            coding_url = CODING_DOWNLOAD_URL_PREFIX + str(coding) + CODING_DOWNLOAD_URL_SUFFIX
            print("Downloading %s" % (coding_url))
            html = urllib.request.urlopen(coding_url)
            contents = html.read().decode(TEXT_ENCODER)
            html.close()
            with open(coding_filename, "w", encoding=TEXT_ENCODER) as file:
                file.write(contents)
        if os.path.exists(coding_filename):
            coding_config[str(coding)] = {}
    with open(coding_config_filename, "w") as coding_config_file:
        coding_config.write(coding_config_file)

def coding_config_generation(coding_config_filename, coding_data_dir):
    coding_config = load_config_from_file(coding_config_filename)
    for coding in coding_config.sections():
        if str(coding) in IRREGULAR_CODING_SET:
            continue
        coding_filename = get_coding_filename(coding_data_dir, str(coding))
        with open(coding_filename, "r", encoding=TEXT_ENCODER) as coding_file:
            coding_html = "".join(coding_file.readlines())
            ind = coding_html.index(TAG_TABLE_START)
            ind_start = coding_html[ind+1:].index(TAG_TABLE_START) + (ind+1)
            ind_end = coding_html[ind_start:].index(TAG_TABLE_END) + ind_start + len(TAG_TABLE_END)
            table = pd.read_html(coding_html[ind_start: ind_end])[0]
            headers = table.columns.values
            assert table.shape[1] >= 2 and HEADER_CODING in headers and HEADER_MEANING in headers and \
                len(table[HEADER_CODING]) == table.shape[0] and len(table[HEADER_MEANING]) == table.shape[0]
            has_selectable = HEADER_SELECTABLE in headers
            non_selectable_set = {}
            for l in range(table.shape[0]):
                if not (has_selectable and str(table[HEADER_SELECTABLE][l]) == TEXT_SELECTABLE_NO):
                    coding_config[str(coding)].update({config_parser_key(str(table[HEADER_CODING][l])): str(table[HEADER_MEANING][l])})
                elif not str(table[HEADER_CODING][l]) in non_selectable_set:
                    non_selectable_set.update({str(table[HEADER_CODING][l]): str(table[HEADER_MEANING][l])})
                else:
                    non_selectable_set.update({str(table[HEADER_CODING][l]): SPECIAL_DESCRIPTION_DUPLICATED})
            for c in non_selectable_set:
                if non_selectable_set[c] != SPECIAL_DESCRIPTION_DUPLICATED:
                    coding_config[str(coding)].update({config_parser_key(c): non_selectable_set[c]})
    with open(coding_config_filename, "w") as coding_config_file:
        coding_config.write(coding_config_file)

def coding_rule_parser(coding_config_filename, raw_data_dir, raw_data_list_filename, coding_data_dir):
    coding_texts_download(coding_config_filename, raw_data_dir, raw_data_list_filename, coding_data_dir)
    coding_config_generation(coding_config_filename, coding_data_dir)
