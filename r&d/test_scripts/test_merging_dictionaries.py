import numpy as np
from collections import Counter

array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
print(array1)
print(array2)
print(type(array1))

test = np.concatenate((array1, array2))
print(test)


dict1 = {'hit_id': np.array([8137, 8068]),
        'particle_id': [684558069757116416, 684558069757116416]}

dict2 = {'hit_id': np.array([10940]),
        'particle_id': [684558069757116416]}

# new_dict = Counter(dict1) + Counter(dict2)
# new_dict = dict(dict1.items() + dict2.items())

new_hit_ids = np.concatenate((dict1['hit_id'], dict2['hit_id']))
new_particle_ids = dict1['particle_id'] + dict2['particle_id']
new_dict = {'hit_id' : new_hit_ids,
            'particle_id' : new_particle_ids}

print("new hit ids:\n", new_hit_ids)
print("new particle ids:\n", new_particle_ids)


print(new_dict)