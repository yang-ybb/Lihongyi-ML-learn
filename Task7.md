```
def calc_ent_grap(x,y):
    """
        calculate ent grap
    """

    base_ent = calc_ent(y)
    condition_ent = calc_condition_ent(x, y)
    ent_grap = base_ent - condition_ent

    return ent_grap
# 香农墒
def calc_shannon_ent(self, data_set):
    num_entries = len(data_set)
    label_counts = {}
    # the number of unique elements and their occurance
    for featVec in data_set:
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
            label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * math.log(prob, 2)
    return shannon_ent
```
