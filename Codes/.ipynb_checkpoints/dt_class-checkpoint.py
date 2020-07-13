## decision tree
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

##########################################################################################

class dec_tree:
    
    def __init__(self, df_class, d=4, product=False):
        self.d = d
        self.ads = df_class
        self.dt_dict = dict()
        self.main(product)
        
    def main(self, product):
        if product != False:
            for i in ['a', 'b']:
                if i not in self.ads.results.keys():
                    self.dt_dict[i] = 'not on platform'
                elif self.ads.results[i] == 'too few samples for Product + Chain':
                    self.dt_dict[i] = 'too few samples for Product + Chain'
                else:
                    self.tree(i, product)
        else:
            for i in ['a', 'b']:
                self.tree(i, product)
        return None
            
    def tree(self, chain, product):
        self.dt_dict[chain] = dict()
        
        model = DecisionTreeRegressor(max_depth=self.d)
        model.fit(self.ads.results[chain]['ads'].copy(), self.ads.results[chain]['y'].copy())

        pred = model.predict(self.ads.results[chain]['ads'])
        model_mape = mape(self.ads.results[chain]['y'], pred)
        
        if product == False:
            dotfile = open(f"tree_{chain}_{self.d}.dot", 'w')
        else:
            dotfile = open(f"tree_{chain}_{product}_{self.d}.dot", 'w')
        tree.export_graphviz(model, out_file = dotfile, feature_names = self.ads.results[chain]['features'])
        dotfile.close()
        
        self.dt_dict[chain]['pred'] = pred
        self.dt_dict[chain]['model'] = model
        self.dt_dict[chain]['dot'] = tree.export_text(model, feature_names = list(self.ads.results[chain]['features']))
        self.dt_dict[chain]['mape'] = model_mape
        return None
    
##########################################################################################

class create_tree:
    
    def __init__(self, clf_obj, chain, product=False):
        clf = clf_obj.dt_dict[chain]['model']
        self.clf = clf
        self.n_nodes = clf.tree_.node_count
        self.children_left = clf.tree_.children_left
        self.children_right = clf.tree_.children_right
        self.feature = clf.tree_.feature
        self.threshold = clf.tree_.threshold
        self.main(chain, product)
        
    def find_path(self, node_numb, path, x):
        path.append(node_numb)
        if node_numb == x:
            return True
        left = False
        right = False
        if (self.children_left[node_numb] !=-1):
            left = self.find_path(self.children_left[node_numb], path, x)
        if (self.children_right[node_numb] !=-1):
            right = self.find_path(self.children_right[node_numb], path, x)
        if left or right :
            return True
        path.remove(node_numb)
        return False
    
    def get_rule(self, path, column_names):
        mask = ''
        for index, node in enumerate(path):
            #We check if we are not in the leaf
            if index!=len(path)-1:
                # Do we go under or over the threshold ?
                if (self.children_left[node] == path[index+1]):
                    mask += "(df['{}']<= {}) \t ".format(column_names[self.feature[node]], self.threshold[node])
                else:
                    mask += "(df['{}']> {}) \t ".format(column_names[self.feature[node]], self.threshold[node])
        # We insert the & at the right places
        mask = mask.replace("\t", "&", mask.count("\t") - 1)
        mask = mask.replace("\t", "")
        return mask
    
    def main(self, chain, product):
        # Leaves
        if product == False:
            leave_id = self.clf.apply(class_attr.results[chain]['ads'])
        else:
            leave_id = self.clf.apply(class_attr_product[product].results[chain]['ads'])

        paths ={}
        for leaf in np.unique(leave_id):
            path_leaf = []
            self.find_path(0, path_leaf, leaf)
            paths[leaf] = np.unique(np.sort(path_leaf))

        rules = {}
        for key in paths:
            if product == False:
                rules[key] = self.get_rule(paths[key], class_attr.results[chain]['ads'].columns)
            else:
                rules[key] = self.get_rule(paths[key], class_attr_product[product].results[chain]['ads'].columns)
            
        self.rules = rules
        return None
    
##########################################################################################

def create_rules(rules, d, product=False):
    temp = pd.DataFrame({'a': list(rules.keys()), 'b': list(rules.values())})
    temp['b'] = temp['b'].str.replace(r'<= 0.5', '=0')
    temp['b'] = temp['b'].str.replace(r'> 0.5', '=1')
    temp['b'] = temp['b'].str.replace("\(df\[", '', regex=True)
    temp['b'] = temp['b'].str.replace("\)|'|\]", '', regex=True)

    temp2 = temp['b'].str.split(' & ', expand=True).rename(columns = lambda x: "b"+str(x+1))
    temp = pd.concat([temp[['a']].copy().reset_index(drop=True), temp2], axis=1)
    if product == False:
        temp.to_csv(f'rules{d}.csv', index=False)
    else:
        temp.to_csv(f'rules{d}_{product}.csv', index=False)
    return temp

##########################################################################################
