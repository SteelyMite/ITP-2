def create_parameter_dictionary(input_type, datatype, text, acceptable_values):
    return {
        "Input": input_type,
        "Datatype": datatype,
        "Text": text,
        "AcceptableValues": acceptable_values
    }

# Example usage:
# SVM_CLASSFICATION = create_parameter_dictionary("dropdown", "Numeric", "Please select a value:", [1, 2, 3, 4, 5])


KMEANS_CLUSTERING = create_parameter_dictionary("text_input", "String", "Enter a text value:", ["foo", "bar", "baz"])

SVM_CLASSFICATION = [{},{},{}]
KMEANS_CLUSTERING = [{},{},{}]

