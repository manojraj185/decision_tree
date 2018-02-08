# Chetan KS, Manoj Rajalbandi, Ajitesh Jayanth
# Decision Tree
from __future__ import print_function

import csv # to read csv
import pandas as pd # to read csv
import math
import operator
import copy

depth = 0
class Node:
    def __init__(self, name, domain):
        self.name = name
        self.child = []
        self.domain = domain

# Dictionary for all the predictors
def create_dict(df):
    Predictors = {}
    for col in df.iloc[:,0:-1]:

        Predictors[col] = {}

        x = df[col].value_counts()
        for i in range(x.size):
            #print (x.index[i])
            Predictors[col][x.index[i]] = {}
            df_attribute = (df.loc[df[col] == x.index[i]])
            count = df_attribute["Enjoy"].value_counts()
            if len(count) == 2:
                Predictors[col][x.index[i]][count.index[0]] = {}
                Predictors[col][x.index[i]][count.index[0]] = count[0]
                Predictors[col][x.index[i]][count.index[1]] = {}
                Predictors[col][x.index[i]][count.index[1]] = count[1]
            elif count.index[0] == "Yes":
                Predictors[col][x.index[i]][count.index[0]] = {}
                Predictors[col][x.index[i]][count.index[0]] = count[0]
                Predictors[col][x.index[i]]["No"] = {}
                Predictors[col][x.index[i]]["No"] = 0
            elif count.index[0] == "No":
                Predictors[col][x.index[i]][count.index[0]] = {}
                Predictors[col][x.index[i]][count.index[0]] = count[0]
                Predictors[col][x.index[i]]["Yes"] = {}
                Predictors[col][x.index[i]]["Yes"] = 0

    return Predictors

def target_entropy(df):
    target_col = df.iloc[:, -1]
    # .iloc[] is primarily integer position based (from 0 to length-1 of the axis), but may also be used with a boolean array.
    # Use iloc and select all rows (:) against the last column (-1):

    X = target_col.value_counts()  # Returns object containing counts of unique values.
    #print(df)
    #print(X)
    if len(X) == 1:
        return 0
    P1 = X["Yes"] / float((X["Yes"] + X["No"]))
    P2 = X["No"] / float((X["Yes"] + X["No"]))
    if P1 == 0 or P2 == 0:
        I = 0
    else:
        I = -(P1 * math.log(P1, 2) + P2 * math.log(P2, 2))
    return I

def entropy(pred_dict, target_entropy):
    info_gain = {}
    entropy_dict = {}
    for col_name in pred_dict:
        total = 0
        I = 0
        for attr in pred_dict[col_name]:
            yes_count = pred_dict[col_name][attr]["Yes"]
            no_count = pred_dict[col_name][attr]["No"]
            yes_no_count = yes_count + no_count
            total += yes_no_count
            P1 = yes_count / float((yes_count + no_count))
            P2 = no_count / float((yes_count + no_count))
            if P1 == 0 or P2 == 0:
                I += 0
            else:
                I += -yes_no_count*(P1 * math.log(P1, 2) + P2 * math.log(P2, 2))
        avg_entopy = I / total
        info_gain[col_name] = target_entropy - avg_entopy
        entropy_dict[col_name] = {avg_entopy}
    return info_gain, entropy_dict
def read_test():
    df = pd.read_csv("test.csv")
    return df
def read_csv():

    df1 = pd.read_csv("dt-data.csv")
    df = copy.deepcopy(df1)
    df.columns = df.columns.str.replace('\W', '')
    df["Occupied"] = df["Occupied"].str.replace('[^a-zA-Z]', '')
    df["Enjoy"] = df["Enjoy"].str.replace('[^a-zA-Z]', '')
    df["Price"] = df["Price"].str.replace('[^a-zA-Z]', '')
    df["Music"] = df["Music"].str.replace('[^a-zA-Z]', '')
    df["Location"] = df["Location"].str.strip()
    df["VIP"] = df["VIP"].str.replace('[^a-zA-Z]', '')
    #df["Favorite Beer"] = df["Favorite Beer"].str.strip('[^a-zA-Z]', '')
    #df["Enjoy"] = df["Enjoy"].apply(lambda x: str(x).replace(";", ""))

    return df
def print_tree(root):
    global depth
    print(root[0].name, end='')
    for i,nd in enumerate(root[0].child):
        print('\n', end='')
        for j in range(0, depth):
            print('\t', end='')
        print(root[0].domain[i] + ':', end='')
        depth += 1
        print_tree([root[0].child[i]])
        depth -= 1
def test_predict(root, df):
    iter = root
    while (iter.name != 'Yes') and (iter.name != 'No'):
        col = iter.name
        val = df[col]
        if val[0] in iter.domain:
            iter = iter.child[iter.domain.index(val[0])]
        else :
            return "Yes"
    return iter.name

def main():
    df = read_csv()
    pred_dict = create_dict(df)
    target_entropy_value = target_entropy(df)
    info_gain, entropy_dict = entropy(pred_dict, target_entropy_value)
    #print("target_entropy_value", target_entropy_value)
    #print("information_gain", info_gain)
    k = max(info_gain.items(), key = operator.itemgetter(1))[0]
    #print("max info gain", k)
    root = Node(k, df[k].value_counts().index.tolist())
    recursion(df, k, [root])
    #print "*****Printing the tree****"
    depth = 0
    print("\n**** The Decision Tree ****\n", end='')
    print_tree([root])
    print("\n**** End of decision tree ****\n", end='')
    dftest = read_test()
    yes_no = test_predict(root,dftest)
    print('\n\n\n**** The label for test data:'+ yes_no + '****\n\n',end='')

def recursion(df, k, root):
    for i,domain in enumerate(df[k].value_counts().index.tolist()):
        df_1 = df.loc[df[k] == domain]
        df_1.set_index(k, inplace = True)
        #print(df_1)
        pred_dict = create_dict(df_1)
        target_entropy_value = target_entropy(df_1)
        if target_entropy_value == 0 or len(df_1.columns) == 2:
            yes_no = df_1.iloc[:,-1]
            root[0].child.append(Node(yes_no[0], []))
            continue
        info_gain, entropy_dict = entropy(pred_dict, target_entropy_value)
        #print("target_entropy_value", target_entropy_value)
        #print("information_gain", info_gain)
        k1 = max(info_gain.items(), key=operator.itemgetter(1))[0]
        #print("max info gain", k1)
        n1 = Node(k1, df_1[k1].value_counts().index.tolist())
        root[0].child.append(n1) 
        recursion(df_1, k1, [root[0].child[i]])

if __name__ == "__main__":
    main()