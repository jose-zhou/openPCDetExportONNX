# class A():
#     def __str__(self):
#         return "aaaa"

# a = A()

# print(a)
import torch
ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(2)]
print(ret_dict)