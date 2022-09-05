# -*- coding: utf-8 -*-

# selected top k gap distance features
def top_k_nearest(gap_dict, k):
  new_gap_dict = {}
  try:
    list1= sorted(gap_dict.values())
  except:
    print(gap_dict)
  top_k_value = list1[:k]
  for obj in gap_dict:
    if gap_dict[obj] in top_k_value:
      new_gap_dict[obj] = gap_dict[obj]
  return new_gap_dict