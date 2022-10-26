import pickle
import numpy as np

file_to_read = open("pred_relations.dict.eval.model.14.bin.2021-07-15-04-15-32", "rb")
predictions = pickle.load(file_to_read)
file_to_read.close()

file_to_read = open("interactive_relations.pickle", "rb")
detections = pickle.load(file_to_read)
file_to_read.close()

confusion_matrix = np.zeros((3, 3))

for scenario_id in detections:
    prediction = predictions[scenario_id]
    detection = detections[scenario_id][-1]
    if prediction == 0:
        if detection == 0:
            confusion_matrix[0, 0] += 1
        elif detection == 1:
            confusion_matrix[1, 0] += 1
        elif detection == 2:
            confusion_matrix[2, 0] += 1
    elif prediction == 1:
        if detection == 0:
            confusion_matrix[0, 1] += 1
        elif detection == 1:
            confusion_matrix[1, 1] += 1
        elif detection == 2:
            confusion_matrix[2, 1] += 1
    elif prediction == 2:
        if detection == 0:
            confusion_matrix[0, 2] += 1
        elif detection == 1:
            confusion_matrix[1, 2] += 1
        elif detection == 2:
            confusion_matrix[2, 2] += 1

print("result: \n", confusion_matrix)
#
# if scenario_id in loaded_dictionary:
#     relation = loaded_dictionary[scenario_id]
#     agent_ids = []
#     for agent_id in agent_dic:
#         if agent_dic[agent_id]['to_predict']:
#             agent_ids.append(agent_id)
#     if relation == 0:
#         edges = [[agent_ids[0], agent_ids[1], 0]]
#     elif relation == 1:
#         edges = [[agent_ids[1], agent_ids[0], 0]]
#     else:
#         edges = []