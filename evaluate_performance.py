import matplotlib.pyplot as plt
import sys
import os 
import numpy as np


def list_of_lists_into_product_of_lists(list_of_lists):
    # input = [[a1,a2], [b1,b2]]
    # output = [[a1,b1],[a1,b2],[a2,b1],[a2,b2]]
    if len(list_of_lists) == 0:
        return []
    if len(list_of_lists) == 1:
        return [[element] for element in list_of_lists[0]]
    else :
        product = []
        sub_product_of_lists = list_of_lists_into_product_of_lists(list_of_lists[1:])
        if len(list_of_lists[0]) == 0:
            product = sub_product_of_lists
        else:
            for element in list_of_lists[0]:
                for my_list in sub_product_of_lists:
                    product.append([element]+my_list)
        return product



def find_all_files_with_keys(list_of_keys, metric):
    # list_of_keys is a list of lists
    all_files = []
    if metric == "show_perf_curves":
        all_files = [sub_path+f for f in os.listdir(sub_path) if f[0:4]=="perf"]
    else:
        all_files = [sub_path+f for f in os.listdir(sub_path) if f[0:4]=="homo"]
    correct_files = []
    for file in all_files:
        correct = False
        for keys in list_of_keys:
            sub_correct = True
            for key in keys:
                if "_"+key+"_" not in file and "_"+key+".txt" not in file:
                    sub_correct = False
            if sub_correct:
                correct = True
        if correct:
            correct_files.append(file)
    return correct_files


def number_of_lines_in_file(file):
    with open(file,"r") as f:
        my_lines = [line for line in f.readlines() if len(line)>1]
        return len(my_lines)

def extract_homology_profiles_from_file(file):
    # returns a list [[3,9,3],[2,1,2],...]
    with open(file,"r") as f:
        my_lines = [line for line in f.readlines() if len(line)>1]
        # 1 4 1|... -> [1,4,1]
        return [[int(i) for i in line.split("|")[0].split()] for line in my_lines]

def max_b_total_in_file(file):
    return max([sum(profile) for profile in extract_homology_profiles_from_file(file)])

def max_b_0_in_file(file):
    return max([profile[0] for profile in extract_homology_profiles_from_file(file)])

def max_b_1_in_file(file):
    return max([profile[1] for profile in extract_homology_profiles_from_file(file)])

def evaluate_files(files,metric):
    if metric == "mean_number_of_lines":
        return np.mean([number_of_lines_in_file(file) for file in files])
    if metric == "total_number_of_lines":
        return sum([number_of_lines_in_file(file) for file in files])
    if metric == "max_b_total":
        return max([max_b_total_in_file(file) for file in files])
    if metric == "max_b_0":
        return max([max_b_0_in_file(file) for file in files])
    if metric == "max_b_1":
        return max([max_b_1_in_file(file) for file in files])
    if metric == "show_perf_curves":
        for file in files:
            with open(os.path.join(sub_path, file), 'r') as f:           
                performance = np.loadtxt(f,dtype=float)
                plt.plot(performance[:,0], performance[:,1],label= file.split("/")[-1])

        plt.legend()
        plt.show()
        return None


def summed_loop(summed_characteristics, inherited_keys,metric):
    # inherited_keys is a list, summed_characteristics is a list of lists
    if len(summed_characteristics)==0:
        files = find_all_files_with_keys([inherited_keys],metric)
    else:
        files = find_all_files_with_keys([char + inherited_keys for char in summed_characteristics],metric)
    return evaluate_files(files,metric)

def compared_loop(summed_characteristics, compared_characteristics,inherited_keys,metric):
    #print(f"Inside compared loop,  inherited keys {inherited_keys} summed_characteristics {summed_characteristics}")
    # summed_characterisitics and compared_characteristics are list of lists
    if metric != "show_perf_curves":
        text = ""
        if metric == "mean_number_of_lines":
            text = "mean number of homology profiles" 
        if metric == "total_number_of_lines":
            text = "total number of homology profiles" 
        if metric == "max_b_total":
            text = "max b_total"
        if metric == "max_b_0":
            text = "max b_0"
        if metric == "max_b_1":
            text = "max b_1"

        if len(compared_characteristics) == 0:
            result = summed_loop(summed_characteristics, inherited_keys,metric)
            print(f"For parameters = {inherited_keys}, the "+text+f" is {result}")
        else:
            print(f"For parameters = {inherited_keys}, the "+text+" is ")
            for char in compared_characteristics:
                result = summed_loop(summed_characteristics,char+inherited_keys,metric)
                print(f"{result} for {char}") 
    else:
        print(f"For parameters = {inherited_keys}, the performance graphs are :")
        total_list_of_chars = []
        for list_of_chars1 in (summed_characteristics if len(summed_characteristics) != 0 else [[]]):
            for list_of_chars2 in (compared_characteristics if len(compared_characteristics) != 0 else [[]]):
                total_list_of_chars.append(list_of_chars1 + list_of_chars2 + inherited_keys)
        files = find_all_files_with_keys(total_list_of_chars,metric)
        evaluate_files(files,metric)

def other_loop(other_characteristics,summed_characteristics, compared_characteristics,metric):
    # The outer loop
    if len(other_characteristics) == 0:
        compared_loop(summed_characteristics,compared_characteristics, [],metric)
    else :
        for char in other_characteristics:
            compared_loop(summed_characteristics,compared_characteristics, char,metric)




for degree in [4,5,6,7]:

    dim = 3
    # compare the number of homology profiles found with different parameters :

    # homologies_exp_1_dim_3_deg_4_obj_bt_sopttime_30_alg_TS_intriang_Trivial.txt
    # obt, sopttime, alg, intriang
    characteristics = {}
    characteristics["obt"] = ["bt","b0","b0pa1"]
    #characteristics["sopttime"] =  ["5"] # ["30","300"] if degree in {4,5} else ["60","600"]
    #characteristics["alg"] = ["TS", "MCTS"]
    characteristics["intriang"] = ["Trivial","Medium","Large"]
    local_path = "/home/charles/Desktop/ML_RAG/Code/Saved_files_exp_0.1"
    #compared_characteristics = ["sopttime","alg", "intriang"]
    compared_characteristics = ["intriang"]
    summed_characteristics = []
    other_characteristics = ["obt"]
    #  metric takes as values "mean_number_of_lines",  "total_number_of_lines", "max_b_total", "max_b_0", "max_b_1", "show_perf_curves"
    metric = "show_perf_curves"



    other_characteristics = list_of_lists_into_product_of_lists([characteristics[key] for key in other_characteristics])
    compared_characteristics = list_of_lists_into_product_of_lists([characteristics[key] for key in compared_characteristics])
    summed_characteristics = list_of_lists_into_product_of_lists([characteristics[key] for key in summed_characteristics])

    #print(other_characteristics)
    #print(f"\n comp char {compared_characteristics}")
    #print(f"\n summed char {summed_characteristics}")


    sub_path = local_path+f"/dim_{dim}_degree_{degree}/"
    print(f"\nDegree {degree}")

    other_loop(other_characteristics,summed_characteristics,compared_characteristics,metric)