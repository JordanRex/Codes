#### ENCODING CLASS ####
#- various types of encoding here
#    - label encoder
#    - binary encoder
#    - base encoder
#    - hashing encoder
#    - ordinal encoder (similar to label, different implementation)
#    - one-hot encoder

from IMPORT_MODULES import *

class categ_encoders():
    
    def  __init__():
        """ return nothing. do nothing. """

    def encoding(train, valid, y_train, y_valid, which=['le', 'be', 'bne', 'ohe', 'he', 'oe']):
        if which=='le':
            train, valid, categorical_names = categ_encoders.labelEncoder(train, valid)
        elif which in ['be', 'bne', 'ohe', 'he', 'oe']:
            train, valid, categorical_names = categ_encoders.ce_encodings(train, valid, y_train, y_valid, which)
        else :
            print('Not supported. Use one of [be, bne, he, oe, ohe]', '\n')
            exit()            
        return train, valid, categorical_names
        
    def labelEncoder(train_df, valid_df):
        print('label encoding is happening ...', '\n')
        cat_columns = train_df.select_dtypes(include=['object']).columns.values
        categorical_names = {}
        for feature in tqdm(cat_columns):
            le = preprocessing.LabelEncoder()
            le.fit(train_df[feature].astype(str))
            train_df[feature] = le.transform(train_df[feature].astype(str))
            valid_df[feature] = valid_df[feature].map(lambda i: 'No Data' if i not in le.classes_ else i)
            le_classes = le.classes_.tolist()
            bisect.insort_left(le_classes, 'No Data')
            le.classes_ = le_classes
            valid_df[feature] = le.transform(valid_df[feature].astype(str))
            categorical_names[feature] = le.classes_
        print('label encoding completed', '\n')
        return train_df, valid_df, categorical_names
        
    def ce_encodings(train_df, valid_df, y_train, y_valid, encoding):
        print(str(encoding) + ' encoding is happening ...', '\n')
        if encoding=='bne':    
            enc=ce.BaseNEncoder(base=3)
        elif encoding=='be':
            enc=ce.BinaryEncoder()
        elif encoding=='he':
            enc=ce.HashingEncoder()
        elif encoding=='oe':
            enc=ce.OrdinalEncoder()
        elif encoding=='ohe':
            enc=ce.BaseNEncoder(base=1)
        enc.fit(train_df)
        train_enc=enc.transform(train_df)
        valid_enc=enc.transform(valid_df)
        print('category encoding completed', '\n')
        categorical_names = {}
        return train_enc, valid_enc, categorical_names