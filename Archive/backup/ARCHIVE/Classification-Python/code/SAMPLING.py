#### SAMPLING ####
#- Oversampling (ADASYN, SMOTE)
#- Undersampling (ENN, RENN, AllKNN)
#- Oversampling and then Undersampling (SMOTE and ENN/TOMEK)

#*it's okay if you have no idea what the above mean. the only thing that is important is to understand why over/undersampling
#is done and why or what ratio between*
#    - why over/under sampling is done in a classification context
#    - what ratio between the 2 classes is important to You in your context
#    - how much information loss (or gain) are you willing to tolerate? (create More data than what you have at hand?)

#""" Explicitly doing sampling. Use with care if going ahead with the CV based approach. Keep ratio low if so (recommended)

#oversampling the minority class using techniques from SMOTE (for oversampling) and ENN/Tomek (for undersampling/cleaning)
#ENN worked out better than Tomek
#added support for undersampling with ENN/RENN/AllKNN """

from IMPORT_MODULES import *

def sampler(X_train, y_train, which='smote_enn', frac=0.75):
    """ which = ['adasyn', smote_tomek', 'smote_enn', 'enn', 'renn', 'allknn'] """
    
    feat_names = X_train.columns.values
    print('Sampling is being done..\n')

    ### OVERSAMPLING (ADASYN) ###
    if which=='adasyn':
        # Apply ADASYN
        ada = ADASYN(random_state=0)
        X_train, y_train = ada.fit_sample(X_train, y_train)

    ### OVERSAMPLING (SMOTE) AND THEN UNDERSAMPLING (ENN/Tomek) ###
    if which=='smote_tomek':
        # Apply SMOTE + Tomek links
        sm = SMOTETomek(random_state=0, ratio=frac)
        X_train, y_train = sm.fit_sample(X_train, y_train)
    if which=='smote_enn':
        # Apply SMOTE + ENN
        smote_enn = SMOTEENN(random_state=0, ratio=frac)
        X_train, y_train = smote_enn.fit_sample(X_train, y_train)

    ### UNDERSAMPLING (ENN/RENN/AllKNN) ###
    if which=='enn':
        # Apply ENN
        enn = EditedNearestNeighbours(random_state=0)
        X_train, y_train = enn.fit_sample(X_train, y_train)
    if which=='renn':
        # Apply RENN
        renn = RepeatedEditedNearestNeighbours(random_state=0)
        X_train, y_train = renn.fit_sample(X_train, y_train)
    if which=='allknn':
        # Apply AllKNN
        allknn = AllKNN(random_state=0)
        X_train, y_train = allknn.fit_sample(X_train, y_train)

    X_train = pd.DataFrame(data=X_train,columns=feat_names)
    print(X_train.shape, y_train.shape, collections.Counter(y_train))
    
    return X_train, y_train