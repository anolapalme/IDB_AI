import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator

########################################################################################
# Make the table containing all the 16s data of interest and the info of each partient #
########################################################################################

#load the files
metadata = pd.read_csv("hmp2_metadata.csv") #note that I changed manually the column External ID -> OTU_ID you could do this by df = df.rename(columns={'oldName1': 'newName1') I should give this to you to make this easier

'''
Project    OTU_ID                          ...                          Age when started smoking How many cigarettes/cigars/etc. do you smoke per day?
0     C3001CSC1_BP    206615                          ...                         NaN                                                NaN
1     C3001CSC2_BP    206614                          ...                         NaN                                                NaN
2     C3002CSC1_BP    206617                          ...                         NaN                                                NaN
3     C3002CSC2_BP    206619                          ...                         NaN                                                NaN
4     C3002CSC3_BP    206616                          ...                         NaN                                                NaN
5     C3002CSC4_BP    206618                          ...                         NaN                                                NaN
6     C3003CSC1_BP    206621                          ...                         NaN                                                NaN
7     C3003CSC2_BP    206622                          ...                         NaN                                                NaN
8     C3003CSC3_BP    206620                          ...                         NaN                                                NaN
9     C3004CSC1_BP    206624                          ...                         NaN                                                NaN
10    C3004CSC2_BP    206623                          ...                         NaN                                                NaN
'''

transposed_taxonomic_profile.to_csv("transposed_taxonomic_profile.csv")#I will also give this to you to limit the useless crap

'''
     OTU_ID  IP8BSoli  UncTepi3  Unc004ii  Unc00re8  Unc018j2  Unc04u81    ...     UncAl534  Unc63248  Unc01505  Unc27462  Unc00bs1  Unc002ze  UncRumi6
0    206646         0         0         0         0         1         0    ...            0         0       481         0         0         0         0
1    224324         0         0         0         0         0         0    ...            0         0        16         0         0         0         0
2    206619         0         0         0         0         0         0    ...            0         0         4         0         0         0         0
3    224326         0         0         0         0         0         0    ...            0         0       117         0         0         0         0
4    206624        17         0         0         0         0         0    ...            0         0         0         0         0         0         0
5    219644         0         0         0         0         0         0    ...            0         0      1350         0         0         0         0
6    214995         0         0         0         0         0         0    ...            0         0        63         0         0         0         0
7    215058         0         0         0         0         0         0    ...            0         0       134         0         0         0         0
8    206750         2         0         0         0         0         3    ...            0         0       589         0         0         0         4
9    206617         0         0         0         0         0         0    ...            0         0         0         0         0         0         0
10   206626         0         0         0         0         0         0    ...            0         0         8         0         0         0         0
11   219638         0         0         0         0         0         0    ...            0         0       198         0         0         0         0
'''

truncated_metadata =  metadata[metadata.OTU_ID.apply(lambda x: x.isnumeric())] #this is used to remove any entries that are non-numerical, these were there for the non-16S samples that we do not want (virology, metabolome, etc)

data_final = transposed_taxonomic_profile.merge(truncated_metadata, on='OTU_ID', how='inner', suffixes=('_1', '_2')) #merge the two tables by matchin the entries in the column OTU_ID


#download the data

#check the summary if the file
data.describe()

#type enter to continue
input("Type enter to continue")

'''
Screen for the most expressed count for IBD patient
'''


data_list = data.apply(lambda x: x.tolist(), axis=1)
dict_samples = {}

for i in range(0, len(data_list)):
    average = int(sum(data_list[i][1:-1])/(len(data_list[i])-1))    
    dict_samples.update({data_list[i][0]:average})

#check the dictionary itself, we will be taking all the non-zero entry    
print(dict_samples)

clnd_dict_samples = {}
#select the entries that are non zeros 
for key,value in clnd_dict_samples.items():
    if value != 0:
        clnd_dict_samples.update({key:value})

#lets see what we got now 
sorted_clnd_dict_samples = sorted(clnd_dict_samples.items(), key=operator.itemgetter(1))
sorted_clnd_dict_samples.reverse()
sorted_rows = pd.Dataframe(sorted_clnd_dict_samples)
list_keys = list(clnd_dict_samples)

"""
make plot with top hits!
"""

for i in data_list:
    for j in list_keys:
        if i[0] == j:
            plt.plot(i[1:-1] ,'ro')
            plt.title(i[0])
            plt.xlabel('Sample')
            plt.ylabel('Counts')
            plt.savefig("{name}.png".format(name=i[0]))

