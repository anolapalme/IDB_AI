import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator

df3 = pd.read_csv("final_table.csv", sep=',') 

#########################################################
#transform non-numerical categorical variable into digit#
#########################################################
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df3['diagnosis'] = le.fit_transform(df3['diagnosis'])

'''
x = input('do you wish to see what the column looks like? (Y/N)')

if x == 'Y':
    print(df3['diagnosis'])
    print('not that all the variables have been converted to a digit allowing the AI to consider them')

if x == 'N':
    print('Okay I get it, so unappreciative ... uhgggg')
'''
###############################
#now lets choose our variables#
###############################

#lets try all the counts 
X_1 = df3.loc[:, 'IP8BSoli':'UncRumi6']

#with the diagnosis as our categorical variable
y = df3.loc[:,'diagnosis']

######################################
# find the strains to use as a marker#
######################################

listVarInterest = ['Unc01505','Unc03433','Unc05nru','Unc057b2','Od8Spla3','Unc00y95','Unc64172','Unc01w4o','GW6Spe33','UncO8895','Unc01qt1','UncG3786','Unc02ee9','Unc04x9p','Unc85953','Unc019wl','Unc01e6u','Unc38399','G7WHuG28','UncO8782','Unc91005','Unc01w0v','Unc36622','Unc91440','Unc91512','Unc05o9h','Unide146','Unc054m4','Unc04f12','Unc48286','Unc02jcp','Unc71175','Unc04zq9','Unc65343','Unc01c0o','Unc02hhf','UncO6361','Unc050mf','Unc053gf','UncC1868','Unc01c0q','Unc02q6j','Unc054vi','FNWNL411','Unc01ie9','S53Therm','Unc00kv7','Unc91094','Unc01t8m','Unc056wq','Unc94755','Unc02ruj','Unc002el','Unc05768','Unc053aw','Unc00z5u','Unc94574','Unc052sp','GWMAdo11','Unc015db','Unc69508','UncB2490','Unc05bd1','Unc02f9r','UncO6120','Unc00a9i','UncO7791','Unc05ssb','F1NS1036']

iteresting_variable = {}

for i in listVarInterest:
    subset = df3[[i,'diagnosis']]
    subset = subset.apply(lambda x: x.tolist(), axis=1)
    #generate means  of each group
    total_0 = 0
    len_0 = 0
    total_1 = 0
    len_1 = 0
    total_2 = 0
    len_2 = 0

    for j in subset:
        if j[1] == 0:
            total_0 += j[0]
            len_0 += 1
        if j[1] == 1:
            total_1 += j[0]
            len_1 += 1
        if j[1] == 2:
            total_2 += j[0]
            len_2 += 1

    print()
    average_0 = total_0/len_0
    average_1 = total_1/len_1
    average_2 = total_2/len_2
    #calculating the difference
    #0-1
    _0_1 = abs(average_0 - average_1)
    _0_2 = abs(average_0 - average_2)
    _1_2 = abs(average_1 - average_2)

    total_diff = _0_1 + _0_2 + _1_2

    iteresting_variable.update({i:total_diff})

sorted_iteresting_variable = sorted(iteresting_variable.items(), key=operator.itemgetter(1))
print(sorted_iteresting_variable)
listKeys = []
for key,values in sorted_iteresting_variable:
    listKeys.append(key)

FinalBacterialSpecies = []

for r in listKeys[-4::]:
    FinalBacterialSpecies.append(r)

print(FinalBacterialSpecies)
X_2 = df3[FinalBacterialSpecies]
    

#######################################
#Separate the training and testing set#
#######################################
from sklearn import model_selection
results = model_selection.train_test_split(X_2, y, test_size = 0.2, shuffle = True)
X_train, X_test, y_train, y_test = results

'''
print("What model do you want to try?")
print("Support vector machine -> https://en.wikipedia.org/wiki/Support_vector_machine")
print("")
y = input("What model do you want to try? (SVM)")

if y == 'SVM':
    #########
    #run svm#
    #########

    from sklearn import model_selection, svm
    from sklearn.metrics import mean_squared_error

    clf = svm.SVC()

    clf.fit(X_train,y_train)

    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    
    train_err = mean_squared_error(y_train, pred_train)
    test_err = mean_squared_error(y_test, pred_test)
    #we will later cycle through and possible plot them one against the other
    print('training error =', train_err, '\ntesting error = ', test_err)
    #confusion matrix
    from sklearn.metrics import confusion_matrix


    preds = clf.predict(X_test)
    tn , fp , fn , tp = confusion_matrix(y_test, preds).ravel()
    print("======== CONFUSION MATRIX ========")
    print("true negative - ",tn)
    print("true positive - ",tp)
    print("false negative -",fn)
    print("false positive -",fp)
    print("")
    print("")
    print("======== MEAN SQUARE ERROR ========")
    print('training error =', train_err, '\ntesting error = ', test_err)
'''
Z = 'DT' 
if Z == 'DT':
    import graphviz
    from sklearn import model_selection
    from sklearn.metrics import confusion_matrix
    from sklearn import model_selection, tree
    
    #training data
    tr_tn_list = []
    tr_tp_list = []
    tr_fn_list = []
    tr_fp_list = []

    #testing data
    te_tn_list = []
    te_tp_list = []
    te_fn_list = []
    te_fp_list = []
    
    for i in range(1,11):
        clf = tree.DecisionTreeClassifier(max_depth = 10)
        clf.fit(X_train, y_train)
        p_train = clf.predict(X_train)
        p_test = clf.predict(X_test)
        
        #plot what we are seeing
        dot_data = tree.export_graphviz(clf)
        graph = graphviz.Source(dot_data)
        graph.render("IBD_Decision_Tree"+str(i))
'''      
        #calculate confusion matrix for training data
        prin(confusion_matrix(y_train, p_train).ravel())

  
    xaxis = [i for i in range(1,11)]
    plt.plot(xaxis,tr_tn_list,'b')
    plt.plot(xaxis,tr_tp_list,'g')
    plt.plot(xaxis,tr_fn_list,'r')
    plt.plot(xaxis,tr_fp_list,'m')
    
    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
    '''
