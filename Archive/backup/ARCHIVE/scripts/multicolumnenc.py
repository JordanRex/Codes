from sklearn.preprocessing import LabelEncoder

class MultiColumnLabelEncoder(LabelEncoder):
    """
    Wraps sklearn LabelEncoder functionality for use on multiple columns of a pandas dataframe
    Adapted from the discussion page = [https://code.i-harness.com/en/q/1753595]
    """
    
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, dframe):
        """
        - Fit label encoder to pandas columns
        - Access individual column classes via indexing `self.all_classes_`
        - Access individual column encoders via indexing `self.all_encoders_`
        """
        # if columns are provided, iterate through and get `classes_`
        if self.columns is not None:
            # ndarray to hold LabelEncoder().classes_ for each
            # column; should match the shape of specified `columns`
            self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                           dtype=object)
            self.all_encoders_ = np.ndarray(shape=self.columns.shape,
                                            dtype=object)
            
            for idx, column in enumerate(self.columns):
                # fit LabelEncoder to get `classes_` for the column
                le = LabelEncoder()
                print(column)
                le.fit(dframe.loc[:, column].astype('str'))
                # append the `classes_` to our ndarray container
                self.all_classes_[idx] = (column,
                                          np.array(le.classes_.tolist(),
                                                  dtype=object))
                # append this column's encoder
                self.all_encoders_[idx] = le
        else:
            # no columns specified; assume all are to be encoded
            self.columns = dframe.iloc[:, :].columns
            self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                           dtype=object)
            for idx, column in enumerate(self.columns):
                le = LabelEncoder()
                le.fit(dframe.loc[:, column].astype('str'))
                self.all_classes_[idx] = (column,
                                          np.array(le.classes_.tolist(),
                                                  dtype=object))
                self.all_encoders_[idx] = le
        return self

    def fit_transform(self, dframe):
        """
        - Fit label encoder and return encoded labels.
        - Access individual column classes via indexing `self.all_classes_`
        - Access individual column encoders via indexing `self.all_encoders_`
        """
        # if columns are provided, iterate through and get `classes_`
        if self.columns is not None:
            # ndarray to hold LabelEncoder().classes_ for each
            # column; should match the shape of specified `columns`
            self.all_classes_ = np.ndarray(shape=self.columns.shape, dtype=object)
            self.all_encoders_ = np.ndarray(shape=self.columns.shape, dtype=object)
            for idx, column in enumerate(self.columns):
                # instantiate LabelEncoder
                le = LabelEncoder()
                # fit and transform labels in the column
                dframe.loc[:, column] =le.fit_transform(dframe.loc[:, column].astype('str'))
                # append the `classes_` to our ndarray container
                self.all_classes_[idx] = (column, np.array(le.classes_.tolist(), dtype=object))
                self.all_encoders_[idx] = le
        else:
            # no columns specified; assume all are to be encoded
            self.columns = dframe.iloc[:, :].columns
            self.all_classes_ = np.ndarray(shape=self.columns.shape, dtype=object)
            for idx, column in enumerate(self.columns):
                le = LabelEncoder()
                dframe.loc[:, column] = le.fit_transform(dframe.loc[:, column].astype('str'))
                self.all_classes_[idx] = (column, np.array(le.classes_.tolist(), dtype=object))
                self.all_encoders_[idx] = le
        return dframe

    def transform(self, dframe):
        """
        Transform labels to normalized encoding.
        """
        if self.columns is not None:
            for idx, column in enumerate(self.columns):
                le = self.all_encoders_[idx]
                dframe.loc[:, column] = dframe.loc[:, column].map(lambda i: 'NEW' if i not in le.classes_ else i)
                le_classes = le.classes_.tolist()
                bisect.insort_left(le_classes, 'NEW')
                le.classes_ = le_classes
                
                self.all_encoders_[idx] = le
                self.all_classes_[idx] = (column, np.array(le_classes_, dtype=object))
                dframe.loc[:, column] = self.all_encoders_[idx].transform(dframe.loc[:, column].astype('str'))
        else:
            self.columns = dframe.iloc[:, :].columns
            for idx, column in enumerate(self.columns):
                le = self.all_encoders_[idx]
                dframe.loc[:, column] = dframe.loc[:, column].map(lambda i: 'NEW' if i not in le.classes_ else i)
                le_classes = le.classes_.tolist()
                bisect.insort_left(le_classes, 'NEW')
                le.classes_ = le_classes
                
                self.all_encoders_[idx] = le
                self.all_classes_[idx] = (column, np.array(le.classes_.tolist(), dtype=object))
                dframe.loc[:, column] = self.all_encoders_[idx].transform(dframe.loc[:, column].astype('str'))
        return dframe

    def inverse_transform(self, dframe):
        """
        Transform labels back to original encoding.
        """
        if self.columns is not None:
            for idx, column in enumerate(self.columns):
                dframe.loc[:, column] = self.all_encoders_[idx].inverse_transform(dframe.loc[:, column].astype('str'))
        else:
            self.columns = dframe.iloc[:, :].columns
            for idx, column in enumerate(self.columns):
                dframe.loc[:, column] = self.all_encoders_[idx].inverse_transform(dframe.loc[:, column].astype('str'))
        return dframe