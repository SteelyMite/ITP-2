"""
This file contains the inputs for each Data Analysis Function.
Each input is represented by a dictionary
Each Data Analysis Method is associated with a list of such dictionaries

"""


def create_parameter_dictionary(input_type, datatype, text, acceptable_values):
    return {
        "Input": input_type,
        "Datatype": datatype,
        "Text": text,
        "AcceptableValues": acceptable_values
    }


# Defining list of inputs for each Data Analysis Function
# KMeans Input Parameters
KMEANS_CLUSTERING = []
KMEANS_CLUSTERING.append(create_parameter_dictionary("column-select", "Numeric", "Select column(s):", None))
KMEANS_CLUSTERING.append(create_parameter_dictionary("integer", "Numeric", "Number of Clusters:", (1,None)))

# SVM Input Parameters 
SVM_CLASSFICATION = []
SVM_CLASSFICATION.append(create_parameter_dictionary("column-select", "Numeric", "Select column(s):", []))
SVM_CLASSFICATION.append(create_parameter_dictionary("column-select", "String", "Select target Column:", []))
SVM_CLASSFICATION.append(create_parameter_dictionary("dropdown", "String", "Select Kernel Type:", ["linear", "poly", "rbf", "sigmoid"]))





