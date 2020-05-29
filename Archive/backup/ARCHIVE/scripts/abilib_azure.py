#!/usr/bin/python
# Filename: abilib_azure.py

import pandas as pd
import numpy as np
from abilib_azure import *
from collections import OrderedDict
import datetime as dt
from calendar import monthrange
from datetime import datetime, timedelta
import random

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report,confusion_matrix,precision_recall_fscore_support
from imblearn.over_sampling import SMOTE

import string
import random
random.seed=1



def find_all_ch(ch,string):
	return [i for i,s in enumerate(string) if ch==s]

def fix_year(string_date):
	try:
		if string_date is not None:
			s=string_date
			idx = find_all_ch('/',s)
			if int(s[idx[-1]+1:])>17:
				y= '19' + s[idx[-1]+1:]
			else:
				y= '20' + s[idx[-1]+1:]
			s=s[:idx[-1]+1] + y
			return s
		return None
	except Exception as e:
		return None

def get_age_from_birth(col,reference_year=None):
	l=[]
	for year in col:
		year = fix_year(year)
		year = dt.datetime.strptime(year,'%m/%d/%Y')
		year = year.year
		if reference_year is None:
			age = dt.datetime.now().year-year
		else:
			age = reference_year-year
		l.append(age)
	return l

def monthdelta(d1, d2):
	delta = 0
	while True:
		mdays = monthrange(d1.year, d1.month)[1]
		d1 += timedelta(days=mdays)
		if d1 <= d2:
			delta += 1
		else:
			break
	return delta

def monthdelta_apply(df,verbose=False):
	try:
		d1 = df[0]
		d2 = df[1]
		delta = 0
		while True:
			mdays = monthrange(d1.year, d1.month)[1]
			d1 += timedelta(days=mdays)
			if d1 <= d2:
				delta += 1
			else:
				break
	# use with df['months_difference'] = df[['date1', 'date2']].apply(monthdelta, axis=1)
		return delta
	except Exception  as e:
		if verbose:
			print str(e)
		return None

def get_month_delta(col,reference_date,verbose=False):
	l=[]
	reference_date = dt.datetime.strptime(reference_date,'%m/%d/%Y')
	for date in col:
		try:
			date = fix_year(date)
			date = dt.datetime.strptime(date,'%m/%d/%Y')
			delta = monthdelta(date,reference_date)
			l.append(delta)
		except Exception as e:
			if verbose:
				print e
				print date
			l.append(None)
	return l

def move_duplicated_to_column(df,key,verbose=False):
	i=0
	ori = df[~df.duplicated(key)]
	dup = df[df.duplicated(key)]
	while dup.shape[0]!=0:
		aux = dup[~dup.duplicated(key)]
		ori=ori.join(aux.set_index(key),on=key,how='left',rsuffix="_"+str(i+2))
		dup = dup[dup.duplicated(key)]
		i+=1
		if verbose:
			print i
	return ori

def join_head_mov(head_filename,path,head_prefix,mov_filename,mov_prefix):
	head = pd.read_csv(path+head_filename, dtype=str)
	head.columns = [head_prefix+k for k in head.columns]
	head['key'] = head[head_prefix+'Personnel number']


	mov = pd.read_csv(path+mov_filename, dtype=str)
	mov = move_duplicated_to_column(mov)
	mov.columns = [mov_prefix+k for k in mov.columns]
	mov['key'] = mov[mov_prefix+'Pers.No.']

	df = head.join(mov.set_index('key'),on='key',rsuffix="_11")
	return df

def prepare_dataset(df,cols_drop):
	#remove information from employees that dont have hire date information - 109 in total
	df = df[~pd.isnull(df['head_original hire date'])]
	#add information about age
	l = get_age_from_birth(df['head_date of birth'],2014)
	df['age'] = [str(k) for k in l]
	
	df = df.drop(cols_drop,axis=1)
	return df

def get_stats(df,verbose=False):
	l=[]
	for col in df.columns:
		if verbose:
			print col

		s=float(len(df))

		nn = float(df[col].count())
		nn_pct = nn/s

		unique = len(df[col].unique())
		unique_pct = round(unique/s,2)

		null = pd.isnull(df[col]).sum()
		null_pct = round(null/s,2)

		vc = df[col].value_counts()
		if len(vc)>0:
			mf = pd.DataFrame(df[col].value_counts()).iloc[0,0]
		else:
			mf=0
		
		if nn!=0:
			mf_pct = round(mf/nn,2)
		else:
			mf_pct = 0.0

		binary = len("{0:b}".format(unique))


		l.append([col,nn,nn_pct,unique,unique_pct,binary,null,null_pct,mf,mf_pct])
	d=pd.DataFrame(l,columns=['col','not_null','not_null_pct','unique_values','unique_pct','binary','null','null_pct','most_frequent','mf_pct'])
	return d

def get_dummies_sl(df,oh=None,lle=None,return_oh=False,return_all=False):
	if oh is None:
		oh=OneHotEncoder()
	Xle = df.copy(deep=True)
	
	lle=[]
	for col in df.columns:
		values = sorted(df[col].unique())
		le=[]
		for i,v in enumerate(values):  le.append((v,i))
		lle.append((col,OrderedDict(le)))
	lle=OrderedDict(lle)

	for col in df.columns:
		l=[]
		for v in df[col]:
			l.append(lle[col][v])
		Xle[col]=l
			
	Xoh = oh.fit_transform(Xle)
	if return_oh:
		return Xoh,oh
	elif return_all:
		return Xoh,oh,Xle,lle
	else:
		return Xoh

def min_coef(weights,indices):    
	l=[]
	for i in range(1,len(indices)):
		if indices[i-1]==indices[i]-1:
			l.append(abs(weights[indices[i]-1]))
		else:
			l.append(abs(sum(weights[indices[i-1]:indices[i]-1])))
	imin = np.argmin(l)
	return imin

def RFE_personalized(aux,y,scoring='accuracy',n_features_left=3,clf=None,imbalance=False,verbose=False):
	l=[]
	oh = OneHotEncoder()
	le = LabelEncoder()
	cols = aux.columns.values

	if clf is None:
		clf=LogisticRegression()

	while len(cols)>n_features_left:
		cols = aux.columns.values
		if verbose:
			print len(cols),
		X,oh=get_dummies_sl(aux,oh=oh,return_oh=True)
		if imbalance:
			sm=SMOTE()
			X, y = sm.fit_sample(X.toarray(), y)
		clf,r=cross_validate_model(X,y,cv=3,scoring=scoring,verbose=False,return_clf=True,clf=clf)

		indices=oh.feature_indices_
		try:
			weights = clf.feature_importances_
		except Exception as e: 
			#print e
			if "sklearn.svm" in str(type(clf)):
				weights = clf.coef_.todense().tolist()[0]
			else:
				weights = clf.coef_[0]
				
		idx = min_coef(indices=indices,weights=weights)
		aux= aux.drop([cols[idx]],axis=1)
		r+=[len(cols),cols[idx],cols]
		l.append(r)

	cols=['model','uniform','stratified','most frequent','n_cols','col rm','cols']
	df=pd.DataFrame(l,columns=cols)
	return df

def convert_dt(s,date_format="%m/%d/%Y"):
	try:
		date = dt.datetime.strptime(s,date_format)
		return date
	except:
		return None


def get_col_date(df):
	cols = [k for k in df.columns if 'date' not in k.lower()]
	l=list(set(df.columns) - set(cols))
	df=df[cols]
	for col in df.columns:
		#print col,
		for i in range(5):
			try:
				vals = df[col].dropna()
				idx=np.random.randint(len(vals))
				val = vals.reset_index(drop=True)[idx]
				ch=find_all_ch("/",val)
				if len(ch)>1:
					#l.append([col,val])
					l.append(col)
					break
			except Exception as e:
				pass
	return l

def get_lenght_col(cd1,cd2):
	cd1 = cd1.apply(fix_year)
	cd1 = cd1.apply(convert_dt)
	
	cd2 = cd2.apply(fix_year)
	cd2 = cd2.apply(convert_dt)
	
	df=pd.DataFrame(cd1)
	df[cd2.name] = cd2
	df.index=cd1.index
	
	length = df.apply(monthdelta_apply,axis=1)
	return length

def convert_dt(s,date_format="%m/%d/%Y"):
	try:
		date = dt.datetime.strptime(s,date_format)
		return date
	except:
		return None 

def roman2latin(a):
	a=(a.replace('-b','.5').replace('-a','.0').replace('ix','9')
	.replace('xi','11').replace('x','10').replace('viii','8')
	.replace('vii','7').replace('vi','6').replace('iv','4').replace('v','5')
	.replace('iii','3').replace('ii','2')
	.replace('i','1')
	.replace('ebm-0','12').replace('cre','12').replace('crb','12')
	.replace('crd','12').replace('crf','12').replace('crc','12')
	.replace('crg','12').replace('cra','12'))
	return a

def get_stats(df,verbose=False):
	l=[]
	for col in df.columns:
		if verbose:
			print col

		s=float(len(df))

		nn = float(df[col].count())
		nn_pct = nn/s

		unique = len(df[col].unique())
		unique_pct = round(unique/s,2)

		null = pd.isnull(df[col]).sum()
		null_pct = round(null/s,2)

		vc = df[col].value_counts()
		if len(vc)>0:
			mf = pd.DataFrame(df[col].value_counts()).iloc[0,0]
		else:
			mf=0
		
		if nn!=0:
			mf_pct = round(mf/nn,2)
		else:
			mf_pct = 0.0

		binary = len("{0:b}".format(unique))


		l.append([col,nn,nn_pct,unique,unique_pct,binary,null,null_pct,mf,mf_pct])
	d=pd.DataFrame(l,columns=['col','not_null','not_null_pct','unique_values','unique_pct','binary','null','null_pct','most_frequent','mf_pct'])
	return d

def save_csv_adls(df,filename,adls):
	'''
	Customize function to save a dataframe to a csv file directly inside an azure datalake
	
	Parameters:
	
	df: dataframe to be saved
	filename: complete path plus filename and extension, where the file will be saved in the datalake
	adls: datalake connection object
	'''
	#create a csv object in the memory
	df_str = df.to_csv(index=False)
	
	#open a object using the path
	with adls.open(filename,'wb') as f:
		#write the csv object into the path object
		f.write(str.encode(df_str))
		#close the path object
		f.close()

def fix_year(string_date):
	try:
		if string_date is not None:
			s=string_date
			idx = find_all_ch('/',s)
			if int(s[idx[-1]+1:])>17:
				y= '19' + s[idx[-1]+1:]
			else:
				y= '20' + s[idx[-1]+1:]
			s=s[:idx[-1]+1] + y
			return s
		return None
	except Exception as e:
		return None

def convert_id(col):
	l=[]
	for k in col:
		k=str(k)
		if len(k)==4: k='1000'+k
		elif len(k)==5: k='100'+k
		elif len(k)==6: k='10'+k
		elif len(k)==7: k='1'+k
		l.append(k)
	return l

def convert_id_apply(k):
	try:
		if len(k)==4: k='1000'+k
		elif len(k)==5: k='100'+k
		elif len(k)==6: k='10'+k
		elif len(k)==7: k='1'+k
		return k
	except Exception as e:
		return k

band_dict = {
			 'ceo':0.0, 'ebm':0.0,
			 '0-a':0.0,'0-b':0.5,
			 'i-a':1.0,'i-b':1.5,
			 'ii-a':2.0,'ii-b':2.5,
			 'iii-a':3.0,'iii-b':3.5,
			 'iv-a':4.0,'iv-b':4.5,
			 'v-a':5.0,'v-b':5.5,
			 'vi-a':6.0,'vi-b':6.5,
			 'vii-a':7.0,'vii-b':7.5,
			 'viii-a':8.0,'viii-b':8.5,
			 'ix-a':9.0,'ix-b':9.5,
			 'x-a':10.0,'x-b':10.5,
			 'xi-a':11.0,'xi-b':11.5,
			 
			 'cra-t1':12.0,'cra-t2':12.0,'cra-t3':12.0,
			 'crb-t1':12.0,'crb-t2':12.0,'crb-t3':12.0,'cr-b':12.0,
			 'crc-t1':12.0,'crc-t2':12.0,'crc-t3':12.0,'crc-t4':12.0,
			 'crd-t1':12.0,'crd-t2':12.0,'crd-t3':12.0,'crd-t4':12.0,
			 'cre':12.0,'cre-t1':12.0,'cre-t2':12.0,'cre-t3':12.0,
			 'crf-t1':12.0,'crf-t2':12.0,'crf-t3':12.0,'crf-t4':12.0,
			 'crg-t1':12.0,'crg-t2':12.0,'crg-t3':12.0,
			 'cr-a':12.0, 'cr-c':12.0, 'cr-d':12.0, 'cr-e':12.0, 'cr-f':12.0,
			 'cr-g':12.0, 'crc':12.0, 'crf':12.0, 'crg':12.0,
			 
			 'usd2':12.0,'usd4':12.0,
			 
			 'qct-1':12.0,'qct-2':12.0,'qct-3':12.0,'qct-4':12.0,
			 'qct-c':12.0,'qct-d':12.0,'qct-i':12.0,'qct-ii':12.0,
			 'qgr-ii':12.0,'qgr-iii':12.0,
			 
			 'cont':12.0,
			 'advsr':2.0,
			 
			 'qppt':12.0,
			 'qrep':12.0,
			 'hourly':12.0,
			}

def replace_special(element,enconding=None,verbose=False):

	def replace_str(text):
		if pd.isnull(text):
			return None
		#if '?' in text: text = text.replace('?',' ')
		try:
			str(text)
			text.encode('utf-8')
			text = text.lower()
		except:
			try:
				text=str(text)
			except:
				if verbose:
					print text
				return "codec error"
			text = text.lower()

			if '\xcc\xe4' in text: text = text.replace('\xcc\xe4','e')
			if '\x87\xc6' in text: text = text.replace('\x87\xc6','ca')
			if '\x8d\x86' in text: text = text.replace('\x8d\x86','ca')
			if '\x8d\x8b' in text: text = text.replace('\x8d\x8b','ca')
			if '\x8d\x9b' in text: text = text.replace('\x8d\x8b','co')
			if '\xe7\xe3' in text: text = text.replace('\xe7\xe3','ca')
			if '\x87\xc6' in text: text = text.replace('\x87\xc6','ca')
			if '\x87\xe4' in text: text = text.replace('\x87\xe4','co')

			if 'ri\x88' in text: text = text.replace('ri\x88','rie')
			if '\xb5re' in text: text = text.replace('\xb5re','are')
			if '\xa3de' in text: text = text.replace('\xa3de','ude')
			if '\xa1st' in text: text = text.replace('\xa1st','ist')
			if '\xa1di' in text: text = text.replace('\xa1di','idi')
			if '\x88nc' in text: text = text.replace('\x88nc','enc')
			if '\xa2lo' in text: text = text.replace('\xa2lo','olo')
			if '\xa3bl' in text: text = text.replace('\xa3bl','ubl')
			if '\xa1si' in text: text = text.replace('\xa1si','isi')
			if '\xa1mi' in text: text = text.replace('\xa1mi','imi')
			if '\xa1de' in text: text = text.replace('\xa1de','ide')
			if '\xc4a' in text: text = text.replace('\xc4a','a')


			if enconding!=None:
				if '\xa1' in text: text = text.replace('\xa1','i')
				if '\xa2' in text: text = text.replace('\xa2','o')


			if '\xb1' in text: text = text.replace('\xb1','n')
			if '\xf0' in text: text = text.replace('\xf0','-')
			if '\x98' in text: text = text.replace('\x98','-')
			if '\xff' in text: text = text.replace('\xff','-')
			if '\xa6' in text: text = text.replace('\xa6','-')
			if '\xbd' in text: text = text.replace('\xbd','-')
			if '\x8c' in text: text = text.replace('\x8c','i')
			if '\xc3' in text: text = text.replace('\xc3','')
			if '\x8d\x8bo' in text: text = text.replace('\x8d\x8bo','cao')
			if '\xca' in text: text = text.replace('\xca','e')
			if '\x90' in text: text = text.replace('\x90','e')
			if '\x92' in text: text = text.replace('\x92','i')
			if '\x97' in text: text = text.replace('\x97','o')
			if '\x87' in text: text = text.replace('\x87','a')
			if '\xee' in text: text = text.replace('\xee','o')
			if '\xd4' in text: text = text.replace('\xd4','o')

			if '\xec' in text: text = text.replace('\xec','u')
			if '\x86' in text: text = text.replace('\x86','a')
			if '\x88' in text: text = text.replace('\x88','a')
			if '\x84' in text: text = text.replace('\x84','o')
			if '\xed' in text: text = text.replace('\xed','i')
			if '\xf3' in text: text = text.replace('\xf3','o')
			if '\xe1' in text: text = text.replace('\xe1','a')
			if '\xf5' in text: text = text.replace('\xf5','o')
			if '\xe9' in text: text = text.replace('\xe9','e')
			if '\xea' in text: text = text.replace('\xea','e')
			if '\xf6' in text: text = text.replace('\xf6','e')
			if '\xe4' in text: text = text.replace('\xe4','e')
			if '\xe8' in text: text = text.replace('\xe8','e')

			if '\xa0' in text: text = text.replace('\xa0','a')
			if '\xa1' in text: text = text.replace('\xa1','a')
			if '\xc1' in text: text = text.replace('\xc1','a')
			if '\xa2' in text: text = text.replace('\xa2','a')
			if '\xa3' in text: text = text.replace('\xa3','a')
			if '\xa4' in text: text = text.replace('\xa4','a')
			if '\xe3' in text: text = text.replace('\xe3','a')
			if '\xe0' in text: text = text.replace('\xe0','a')

			if '\x82' in text: text = text.replace('\x82','a')
			if '\x83' in text: text = text.replace('\x83','a')
			if '\x81' in text: text = text.replace('\x81','a')
			if '\x80' in text: text = text.replace('\x80','a')

			if '\xa9' in text: text = text.replace('\xa9','e')
			if '\xaa' in text: text = text.replace('\xaa','e')

			if '\x89' in text: text = text.replace('\x89','e')
			if '\x8a' in text: text = text.replace('\x8a','e')
			if '\x8e' in text: text = text.replace('\x8e','e')

			if '\xad' in text: text = text.replace('\xad','i')

			if '\x8d' in text: text = text.replace('\x8d','i')
			if '\xcd' in text: text = text.replace('\xcd','i')

			if '\xb3' in text: text = text.replace('\xb3','o')
			if '\xb4' in text: text = text.replace('\xb4','o')
			if '\xb5' in text: text = text.replace('\xb5','o')
			if '\xd5' in text: text = text.replace('\xd5','o')
			if '\xc5' in text: text = text.replace('\xc5','o')

			if '\x93' in text: text = text.replace('\x93','o')
			if '\x94' in text: text = text.replace('\x94','o')
			if '\x95' in text: text = text.replace('\x95','o')

			if '\xba' in text: text = text.replace('\xba','u')
			if '\xda' in text: text = text.replace('\xda','u')
			if '\xbc' in text: text = text.replace('\xbc','u')
			if '\xfc' in text: text = text.replace('\xfc','u')

			if '\x9a' in text: text = text.replace('\x9a','u')
			if '\x9c' in text: text = text.replace('\x9c','u')
			if '\xfa' in text: text = text.replace('\xfa','u')

			if '\xa7' in text: text = text.replace('\xa7','c')
			if '\x87' in text: text = text.replace('\x87','c')
			if '\xe7' in text: text = text.replace('\xe7','c')
			if '\x8d' in text: text = text.replace('\x8d','c')
			if '\xc7' in text: text = text.replace('\xc7','c')

			if '\xf4' in text: text = text.replace('\xf4','o')
			if '\xe2' in text: text = text.replace('\xe2','a')
			if '\xb0' in text: text = text.replace('\xb0','o')
			if '\xb2' in text: text = text.replace('\xb2','q')
			if '\xc9' in text: text = text.replace('\xc9','e')
			if '\xd3' in text: text = text.replace('\xd3','o')
			if '\xeb' in text: text = text.replace('\xeb','e')
			if '\xc8' in text: text = text.replace('\xc8','e')
			if '\xf1' in text: text = text.replace('\xf1','n')
			if '\xc2' in text: text = text.replace('\xc2','a')
			if '\xc0' in text: text = text.replace('\xc0','a')
			if '\xdf' in text: text = text.replace('\xdf','s')
			if '\xae' in text: text = text.replace('\xae','i')
			if '\xb7' in text: text = text.replace('\xb7','-')
			if '\xf9' in text: text = text.replace('\xf9','u')
			if '\xac' in text: text = text.replace('\xac','o')
			if '\xf2' in text: text = text.replace('\xf2','o')
			if '\xd6' in text: text = text.replace('\xd6','-')
			if '\xb9' in text: text = text.replace('\xb9','-')

			if '\xf8' in text: text = text.replace('\xf8','o')
			if '\xce' in text: text = text.replace('\xce','i')
			if '\xdc' in text: text = text.replace('\xdc','u')
			if '\xa8' in text: text = text.replace('\xa8','-')
			if '\x8b' in text: text = text.replace('\x8b','a')
			if '\x99' in text: text = text.replace('\x99','o')
			if '\x9f' in text: text = text.replace('\x9f','u')
			if '\xe5' in text: text = text.replace('\xe5','a')
			if '\x9b' in text: text = text.replace('\x9b','o')
			if '\xe6' in text: text = text.replace('\xe6','e')
			if '\xcc' in text: text = text.replace('\xcc','a')
			if '\x91' in text: text = text.replace('\x91','e')

			if '\xef' in text: text = text.replace('\xef','o')
			if '\x96' in text: text = text.replace('\x96','n')
			if '\xcb' in text: text = text.replace('\xcb','a')
			if '\x8f' in text: text = text.replace('\x8f','e')
			if '\xbb' in text: text = text.replace('\xbb','-')
			if '\x9d' in text: text = text.replace('\x9d','u')
			if '\xd2' in text: text = text.replace('\xd2','-')
			if '\xab' in text: text = text.replace('\xab','-')
			if '\xbf' in text: text = text.replace('\xbf','o')
			if '\xd0' in text: text = text.replace('\xd0','-')
			if '\xdb' in text: text = text.replace('\xdb','S')
			if '\xd1' in text: text = text.replace('\xd1','-')
			if '\xfb' in text: text = text.replace('\xfb','O')
			if '\xc6' in text: text = text.replace('\xc6','a')
			if '\xb6' in text: text = text.replace('\xb6','a')
			if '\x85' in text: text = text.replace('\x85','a')

		try:
			str(text)
		except:
			print text
		return str(text)


	def replace_list(l):
		s=[]
		for x in l:
			s.append(replace_str(x))
		return s

	def replace_df(df):
		for column in df.columns:
			df[column] = replace_list(df[column])
		col = df.columns
		col = replace_list(col)
		df.columns = col
		return df

	if type(element) is str or type(element) is unicode: 
		element = replace_str(element)
	elif type(element) is list or 'pandas.core.series.Series' in str(type(element)):
		element = replace_list(element)
	elif 'pandas.core.frame.DataFrame' in  str(type(element)):
		element = replace_df(element)
	else: 
		print('Please use string, unicode, list or pandas Dataframe')

	return element


def get_aggregates(df,cols,cols_agg):
	for i in range(len(cols)):
		col=cols[i]
		col_agg =cols_agg[i]
		new_col = cols_agg[i] +' by '+cols[i]
		df=get_mean_by_group(df,col,col_agg,new_col)
		df[new_col] = get_stds(df[new_col])
	return df

def get_mean_by_group(df,col,col_agg,new_col):
	aux_dict = df.groupby([col]).agg({col_agg:'mean'}).to_dict()[col_agg]
	df[new_col] = 0
	for k in aux_dict.keys():
		f=(df[col]==k)
		idx = df[f].index
		df.ix[idx,new_col] = df.ix[idx,col_agg]/aux_dict[k]
	return df
	
def get_stds(col,label = ['very below','below','mean','above','very above']):
	sm=col.mean()
	st=col.std()
	stds = [sm-2*st,sm-st,sm+st,sm+2*st]
	l=[]
	for v in col:
		if v<stds[0]: l.append(label[0])
		elif v<stds[1]: l.append(label[1])
		elif v<stds[2]: l.append(label[2])
		elif v<stds[3]: l.append(label[3])
		else: l.append(label[4])
	return l

def lleMap(x,le):
	try:
		return le[x]
	except:
		return le['unknown']

def labelEncoder(df,lle=None):
	if lle is None:
		lle=[]
		n=len(df.columns)
		for j,col in enumerate(df.columns):
			values = sorted(df[col].unique())
			values.append('unknown')
			le=[]
			for i,v in enumerate(values):  le.append((v,int(i)))
			lle.append((col,OrderedDict(le)))
		lle=OrderedDict(lle)

	Xle = df.copy(deep=True)

	for j,col in enumerate(Xle.columns):
		le = lle[col]
		Xle[col] = Xle[col].apply(lleMap,args=(le,))
	return Xle,lle


def binaryEncoder(Xle,lle):
	Xoh = None
	for col in Xle.columns:
		m=len(lle[col])
		ll=[]
		try:
			for i,v in enumerate(Xle[col]):
				l=np.zeros(m)
				l[v]=1
				ll.append(l)
		except:
			print col
			pass
		dfcol=pd.DataFrame(ll)
		dfcol.columns = [col+'_'+str(k) for k in lle[col].keys()]
		if Xoh is None:
			Xoh = dfcol
		else:
			Xoh = Xoh.join(dfcol)
	return Xoh

def webService_transposeOpr(df):
	df['Global ID'] = convert_id(df['Employee Global ID'])
	cols_remove = ['Personnel Number', 'First Name', 'Last Name','Zone Name',
				   'Final Evaluation Workflow State','Review Period','Employee Global ID']
	df=df.drop(cols_remove,axis=1)


	#special modifications for opr
	f=(df['OPR Rating Scale']=='1A') | (df['OPR Rating Scale']=='1B')
	#df=df[~f].reset_index(drop=True)
	df=df[df['Year']!=2016]
	p=df[['Year','OPR Rating Scale','Global ID']]
	p=p.pivot(index='Global ID',columns='Year',values='OPR Rating Scale').reset_index().set_index('Global ID')
	p=p[[k for k in p.columns if k!='Year']]
	p.columns = ['opr '+str(k) for k in p.columns]

	df=df[~df.duplicated('Global ID')].drop(['Year','OPR Rating Scale','Plan Name'],axis=1)
	df=df.join(p,on='Global ID',how='left')
	opr_col=[k for k in df.columns if 'opr' in str(k)]
	opr=df.copy(deep=True)
	return opr

def webService_preProcessOprDataset(oprTransposed):
	opr=oprTransposed.copy(deep=True)
	df =oprTransposed.copy(deep=True) 

	l=[]
	for k in opr['Position Title']:
		if 'dir' in str(k).lower():
			l.append('director')
		elif 'manager' in str(k).lower() or 'mgr' in str(k).lower() or 'gm,' in str(k).lower():
			l.append('manager')
		elif 'spec' in str(k).lower():
			l.append('specialist')
		elif 'engineer' in str(k).lower():
			l.append('engineer')
		elif 'vp,' in str(k).lower() or 'vp.' in str(k).lower() or 'vice president' in str(k).lower() or 'vp ' in str(k).lower():
			l.append('vice president')
		elif 'analyst' in str(k).lower():
			l.append('analyst')
		elif 'coordinator' in str(k).lower():
			l.append('coordinator')
		elif 'supervisor' in str(k).lower():
			l.append('supervisor')
		elif 'sales' in str(k).lower() or 'commercial' in str(k).lower():
			l.append('comercial')
		elif 'admin' in str(k).lower() or 'planner' in str(k).lower():
			l.append('administrative')
		elif 'brewmaster' in str(k).lower():
			l.append('brewmaster')
		elif 'maint' in str(k).lower():
			l.append('maintenance')
		elif 'associate' in str(k).lower():
			l.append('associate')
		elif 'temporary' in str(k).lower():
			l.append('temporary')
		elif 'controller' in str(k).lower():
			l.append('controller')
		elif 'merch' in str(k).lower():
			l.append('merchand')
		elif 'tech' in str(k).lower():
			l.append('technician')
		elif 'bpm' in str(k).lower():
			l.append('bpm')
		elif 'legal' in str(k).lower() or 'counsel' in str(k).lower():
			l.append('legal')
		elif 'fsem' in str(k).lower():
			l.append('fsem')
		elif 'accountant' in str(k).lower():
			l.append('accountant')
		elif 'assoc' in str(k).lower():
			l.append('associate')
		elif 'chef' in str(k).lower():
			l.append('chef')
		elif 'coord' in str(k).lower():
			l.append('coordinator')
		elif 'supv' in str(k).lower():
			l.append('supervisor')
		elif 'lead' in str(k).lower():
			l.append('lead')
		elif 'rep.' in str(k).lower():
			l.append('rep')
		elif 'president' in str(k).lower():
			l.append('president')
		elif 'head' in str(k).lower():
			l.append('head')
		elif 'architect' in str(k).lower():
			l.append('architect')
		elif 'ppm' in str(k).lower():
			l.append('ppm')
		elif 'personnel' in str(k).lower():
			l.append('personnel')
		elif 'no name available' in str(k).lower():
			l.append('not available')
		elif 'attendant' in str(k).lower():
			l.append('attendant')
		elif 'driver' in str(k).lower():
			l.append('driver')
		elif 'educa' in str(k).lower():
			l.append('education')
		elif 'intern' in str(k).lower():
			l.append('intern')
		elif 'foreman' in str(k).lower():
			l.append('foreman')
		elif 'sme' in str(k).lower():
			l.append('sme')
		elif 'spm' in str(k).lower():
			l.append('spm')  
		else:
			l.append(str(k))
	df['Position Title'] = l

	df=df.replace(['1A','1B','3A','3B','4A','4B','Not Rated','2'],[1.5,1.0,3.5,3,4.5,4,None,None])
	opr_ori = df[opr_col].copy(deep=True)
	df=df[~opr_ori.isnull().all(axis=1)]

	df['OPR mean abs'] = pd.cut(opr_ori.mean(axis=1),bins=[k/10.0 for k in range(10,60,5)]).astype(str)
	df['OPR max abs'] = pd.cut(opr_ori.max(axis=1),bins=[k/10.0 for k in range(10,60,5)]).astype(str)
	df['OPR min abs'] = pd.cut(opr_ori.min(axis=1),bins=[k/10.0 for k in range(10,60,5)]).astype(str)

	df['OPR mean'] = opr_ori.mean(axis=1)
	df['OPR max'] = opr_ori.max(axis=1)
	df['OPR min'] = opr_ori.min(axis=1)
	df['OPR std'] = opr_ori.std(axis=1)


	cols = ['Position Title']*4 +  ['Function Name Level 2']*4
	cols_agg = ['OPR mean','OPR min','OPR max','OPR std']*2
	df=get_aggregates(df,cols,cols_agg)

	for col in ['OPR mean','OPR min','OPR max','OPR std']:   
		df[col] = get_stds(df[col])

	opr=df.copy(deep=True)
	
	opr.columns = replace_special(list(opr.columns))
	opr.columns = ['opr --- '+k if k!='global id' and k!='label' else k for k in opr.columns]
	opr = replace_special(opr)
	opr = opr.fillna('empty')
	return opr

def get_stds(col,label = ['very below','below','mean','above','very above']):
	sm=col.mean()
	st=col.std()
	stds = [sm-2*st,sm-st,sm+st,sm+2*st]
	l=[]
	for v in col:
		if v<stds[0]: l.append(label[0])
		elif v<stds[1]: l.append(label[1])
		elif v<stds[2]: l.append(label[2])
		elif v<stds[3]: l.append(label[3])
		else: l.append(label[4])
	return l

def webService_addLabel(df,nextYear):
	#initiate the label column
	df['label'] = 0
	#get the ids for the people who were not in the company
	dfrk = set(df['global id'])
	idxOld=list(dfrk.intersection(nextYear))
	#update the labels for the people not in the company
	df = df.set_index('global id')
	df.ix[idxOld,'label'] = 1
	df = df.reset_index()
	return df

def webService_buildDatasetForModel(df,categorical=None,numerical=None,label_name=None,id_name='global id',lle=None):
	df = df.reset_index(drop=True)
	if label_name:
		X = df.drop([label_name,id_name],axis=1)
	else:
		X = df.drop([id_name],axis=1)
	oh = None
	
	if categorical is not None:
		x_categorical = X[categorical]
		Xle,lle = labelEncoder(x_categorical,lle=lle)
		x_categorical = binaryEncoder(Xle,lle)
	if numerical is not None:    
		x_numerical = df[numerical].reset_index(drop=True)
		
	if categorical is not None and numerical is not None:
		X = pd.concat([x_categorical,x_numerical],ignore_index=True,axis=1)
	elif numerical is not None:
		X = x_numerical
	elif categorical is not None:
		X = x_categorical
	 
	dataset = X.copy(deep=True)
	
	if label_name:
		dataset['label']=df[label_name].astype(float).values
	
	return dataset,lle

def webService_trainModel(df):
	y=df['label'].values
	X = df.drop(['label'],axis=1)

	sm=SMOTE() 
	Xs,ys = sm.fit_sample(X,y)
	xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=.3,random_state=1,stratify=y)
	xstrain,xstest,ystrain,ystest = train_test_split(Xs,ys,test_size=.3,random_state=1,stratify=ys)
	xtrain = xstrain
	ytrain = ystrain

	clf=LogisticRegression()
	print cross_val_score(clf,X,y,cv=10,scoring='f1')
	print cross_val_score(clf,Xs,ys,cv=10,scoring='f1')

	clf.fit(xtrain,ytrain)
	pred = clf.predict(xtest)
	print '----'
	print precision_recall_fscore_support(ytest,pred,average='weighted')
	clf.fit(Xs,ys)
	return clf

def get_age_from_birth(col,reference_year=None,fixYear=False):
	l=[]
	for year in col:
		if fixYear:
			year = fix_year(year)
		year = dt.datetime.strptime(year,'%m/%d/%Y')
		year = year.year
		if reference_year is None:
			age = dt.datetime.now().year-year
		else:
			age = reference_year-year
		l.append(age)
	return l

def webService_preProcessOprDataset(df):
	df['Global ID'] = convert_id(df['Employee Global ID'])
	cols_remove = ['Personnel Number', 'First Name', 'Last Name','Zone Name',
				   'Final Evaluation Workflow State','Review Period','Employee Global ID']
	df=df.drop(cols_remove,axis=1)


	#special modifications for opr
	f=(df['OPR Rating Scale']=='1A') | (df['OPR Rating Scale']=='1B')
	#df=df[~f].reset_index(drop=True)
	df=df[df['Year']!=2016]
	p=df[['Year','OPR Rating Scale','Global ID']]
	p=p.pivot(index='Global ID',columns='Year',values='OPR Rating Scale').reset_index().set_index('Global ID')
	p=p[[k for k in p.columns if k!='Year']]
	p.columns = ['opr '+str(k) for k in p.columns]

	df=df[~df.duplicated('Global ID')].drop(['Year','OPR Rating Scale','Plan Name'],axis=1)
	df=df.join(p,on='Global ID',how='left')
	opr_col=[k for k in df.columns if 'opr' in str(k)]
	opr=df.copy(deep=True)

	l=[]
	for k in opr['Position Title']:
		if 'dir' in str(k).lower():
			l.append('director')
		elif 'manager' in str(k).lower() or 'mgr' in str(k).lower() or 'gm,' in str(k).lower():
			l.append('manager')
		elif 'spec' in str(k).lower():
			l.append('specialist')
		elif 'engineer' in str(k).lower():
			l.append('engineer')
		elif 'vp,' in str(k).lower() or 'vp.' in str(k).lower() or 'vice president' in str(k).lower() or 'vp ' in str(k).lower():
			l.append('vice president')
		elif 'analyst' in str(k).lower():
			l.append('analyst')
		elif 'coordinator' in str(k).lower():
			l.append('coordinator')
		elif 'supervisor' in str(k).lower():
			l.append('supervisor')
		elif 'sales' in str(k).lower() or 'commercial' in str(k).lower():
			l.append('comercial')
		elif 'admin' in str(k).lower() or 'planner' in str(k).lower():
			l.append('administrative')
		elif 'brewmaster' in str(k).lower():
			l.append('brewmaster')
		elif 'maint' in str(k).lower():
			l.append('maintenance')
		elif 'associate' in str(k).lower():
			l.append('associate')
		elif 'temporary' in str(k).lower():
			l.append('temporary')
		elif 'controller' in str(k).lower():
			l.append('controller')
		elif 'merch' in str(k).lower():
			l.append('merchand')
		elif 'tech' in str(k).lower():
			l.append('technician')
		elif 'bpm' in str(k).lower():
			l.append('bpm')
		elif 'legal' in str(k).lower() or 'counsel' in str(k).lower():
			l.append('legal')
		elif 'fsem' in str(k).lower():
			l.append('fsem')
		elif 'accountant' in str(k).lower():
			l.append('accountant')
		elif 'assoc' in str(k).lower():
			l.append('associate')
		elif 'chef' in str(k).lower():
			l.append('chef')
		elif 'coord' in str(k).lower():
			l.append('coordinator')
		elif 'supv' in str(k).lower():
			l.append('supervisor')
		elif 'lead' in str(k).lower():
			l.append('lead')
		elif 'rep.' in str(k).lower():
			l.append('rep')
		elif 'president' in str(k).lower():
			l.append('president')
		elif 'head' in str(k).lower():
			l.append('head')
		elif 'architect' in str(k).lower():
			l.append('architect')
		elif 'ppm' in str(k).lower():
			l.append('ppm')
		elif 'personnel' in str(k).lower():
			l.append('personnel')
		elif 'no name available' in str(k).lower():
			l.append('not available')
		elif 'attendant' in str(k).lower():
			l.append('attendant')
		elif 'driver' in str(k).lower():
			l.append('driver')
		elif 'educa' in str(k).lower():
			l.append('education')
		elif 'intern' in str(k).lower():
			l.append('intern')
		elif 'foreman' in str(k).lower():
			l.append('foreman')
		elif 'sme' in str(k).lower():
			l.append('sme')
		elif 'spm' in str(k).lower():
			l.append('spm')  
		else:
			l.append(str(k))
	df['Position Title'] = l

	df=df.replace(['1A','1B','3A','3B','4A','4B','Not Rated','2'],[1.5,1.0,3.5,3,4.5,4,None,None])
	opr_ori = df[opr_col].copy(deep=True)
	df=df[~opr_ori.isnull().all(axis=1)]

	df['OPR mean abs'] = pd.cut(opr_ori.mean(axis=1),bins=[k/10.0 for k in range(10,60,5)]).astype(str)
	df['OPR max abs'] = pd.cut(opr_ori.max(axis=1),bins=[k/10.0 for k in range(10,60,5)]).astype(str)
	df['OPR min abs'] = pd.cut(opr_ori.min(axis=1),bins=[k/10.0 for k in range(10,60,5)]).astype(str)

	df['OPR mean'] = opr_ori.mean(axis=1)
	df['OPR max'] = opr_ori.max(axis=1)
	df['OPR min'] = opr_ori.min(axis=1)
	df['OPR std'] = opr_ori.std(axis=1)


	cols = ['Position Title']*4 +  ['Function Name Level 2']*4
	cols_agg = ['OPR mean','OPR min','OPR max','OPR std']*2
	df=get_aggregates(df,cols,cols_agg)

	for col in ['OPR mean','OPR min','OPR max','OPR std']:   
		df[col] = get_stds(df[col])

	opr=df.copy(deep=True)
	
	opr.columns = replace_special(list(opr.columns))
	opr.columns = ['opr --- '+k if k!='global id' and k!='label' else k for k in opr.columns]
	opr = replace_special(opr)
	opr = opr.fillna('empty')
	return opr

def webService_getModelWeights(clf,lle=None,numerical=[],func_apply='mean'):
	l1=[]
	l2=[]
	if lle:
		for k1 in lle.keys():
			for k2 in lle[k1].keys():
				l1.append(k1)
				l2.append(k2)
	l1+=numerical
	l2+=numerical
	
	gen=pd.DataFrame([l1,l2,list(clf.coef_[0])]).T
	gen.columns = ['variable','value','weight']
	gen['weight']=gen['weight'].astype(float)
	gen['dataset'] = [k[:k.index(' --- ')] if ' --- ' in k else k for k in gen['variable']]
	gen['variable'] = [k[k.index(' --- ')+5:] if ' --- ' in k else k for k in gen['variable']]
	gen['value'] = [k[k.index(' --- ')+5:] if ' --- ' in str(k) else k for k in gen['value']]
	
	return gen

def headcount_procedure(hdf,idx_left=None,fixYear=True):
	'''
	Convert the raw dataset for headcount into a processed dataset to be inserted in the model
	
	inputs:
		hdf = raw headcount dataset
		idx_left = indexes for people who left the company
	'''

	#remove columns that don't have important information or are already represented by others
	cols_remove = ['Original Hire Date', 'Rehire Date','New Hire', 'Personnel number', 'Full Name',
			   'Position','CostCenter Description','ABInbev Entity1',
			   'ABInbev Entity3', 'ABInbev Entity4','Company Code','Personnel area',
			   'Personnel subarea','Employee subgroup','Short Text of Organizational Unit','Gender', 'Ethnicity',
			   'Personnel Area Text','Functional Area','Manager Position','Cost Center']
	hdf=hdf.drop(cols_remove,axis=1)

	#remove special characters and transform all to lower case
	hdf.columns = replace_special(list(hdf.columns))

	#remove duplicated rows from the dataframe
	hdf=hdf[~hdf.duplicated('global id')].reset_index(drop=True)

	#calculate age fom date of birth
	hdf['age'] = get_age_from_birth(hdf['date of birth'],reference_year=2018)
	hdf = hdf.drop(['date of birth'],axis=1)

	#transform age information into percentiles
	hdf['age compared'] = get_stds(hdf['age']) 

	#transform the age information into buckets with 5 years
	hdf['age buckets'] = pd.cut(hdf['age'],bins=range(0,100,5),include_lowest=True).astype(str)
	hdf=hdf.drop(['age'],axis=1)

	#calculate the service year anniversaries based on the service year metric
	for num in [1,2,3,5,10,20]:
		hdf[str(num) + ' - anniversary'] = (hdf['service years']>=num).astype(float)

	#transform service years information into percentiles
	hdf['service years compared'] = get_stds(hdf['service years']) 

	#transform the service years information into buckets with 5 years
	hdf['service years buckets'] = pd.cut(hdf['service years'],bins=range(0,100,5),include_lowest=True).astype(str)

	#convert the global id for the manager to the defined standard 
	hdf['personnel number manager'] = convert_id(hdf['personnel number manager'])

	#add information about pay scale and service years for the manager

	##make a copy of the original dataframe
	aux=hdf.copy(deep=True)
	l=[]

	##get a list of all the possible global ids
	gid = list(aux['global id'].astype(str))

	##loop through all the ids contained in the personal manager column
	for k in aux['personnel number manager'].values:
		try:
			###check if the global id for the manager is in the general list of global ids and return the index
			idx = gid.index(str(k))
			a=idx
		except Exception as e:
			###if the manager global id is not present, just return None
			a=None
		l.append(a)

	##create a new series that contain the indexes for the manager global ids that are presented in the original one
	l=pd.Series(l)

	##slice the dataframe to retrive the important information
	aux=aux.ix[l,['service years','pay scale group']]

	##identify which columns belong to the managers
	aux.columns = ['manager - '+k for k in aux.columns]

	##add the original employee global id information in order to perform the join operation
	aux['global id']=hdf['global id'].values

	##add the manager calculated information to the main dataframe
	hdf=hdf.join(aux.set_index('global id'),on='global id')

	#calculate the gap difference between employee pay scale group and manager pay scale group

	##slice the dataframe to retrieve the information about pay scale group
	aux = hdf[['pay scale group','manager - pay scale group']].copy(deep=True)

	##iterate over each one of the columns in the sliced dataframe
	for col in aux.columns:
		###preprocess the information in each element of the pay scale group
		aux[col+' numerical'] = [k.strip().lower().replace(' ','-').replace('--','-').replace('us-','') if k is not np.nan else k for k in aux[col]]

		###convert each value in pay scale group to a numerical band
		aux[col+' numerical'] = [band_dict[k] if k is not np.nan else k for k in aux[col+' numerical']]

	##calculate the gap in band using the information calculated from pay scale group
	aux['delta band with manager'] = aux['pay scale group numerical']  - aux['manager - pay scale group numerical']

	l=[]
	##convert the delta to defined buckets in order to categorize the information
	for val in aux['delta band with manager']:
		if val<-4:
			a='bigger than -4'
		elif val<-2:
			a='between -4 and -2'
		elif val<0:
			a='between -2 and 0'
		elif val<2:
			a='between 0 and 2'
		elif val<=4:
			a='between 2 and 4'
		elif val>4:
			a='bigger than 4'
		l.append(a)

	##create a new column with the information about gap converted in ordinal categories
	aux['delta band with manager categorized'] = l

	##copy the calculated information into the original dataframe
	hdf['delta band with manager'] = aux['delta band with manager'].values

	#add the information about the label
	if idx_left:
		##initiate the label column
		hdf['label'] = 0

		##update the labels for the people who were not in the company
		hdf = hdf.set_index('global id')

		##assign 1 for people who left the company and reset index
		hdf.ix[idx_left,'label'] = 1
		hdf = hdf.reset_index()
	
	hdf['global id'] = hdf['global id'].astype(float).astype(int).astype(str)
	return hdf

def movements_procedure(mdf,idx_left=None):
	#remove columns that don't have important information or are already represented by others
	cols_remove = ['Pers.No.', 'Personnel Number', 'S', 'Employment Status', 'CoCd','EEGrp','ESgrp',
				   'PA', 'PSubarea', 'AB-InBev Entity.1','InBev Ent L1', 'Macro Entity', 'Mac.Org. Ent. L1',
				   'ActR','Act.','Action Type','Changed by', 'Chngd on','Mac.Org. Ent. L3','Mac.Org. Ent. L2',
				   'InBev Ent L2','InBev Ent L3','InBev Ent L4','Local Entity', 'Local Entity.1', 'Local Entity ID',
				   'Local Entity L2', 'Local Entity L3','AB-I Entity ID','Macro Organizational Entity','Position',
				   'Grp',"Manager's Position",'Start Date.1','Start Date','End Date','Entry','compare','Leaving']
	mdf = mdf.drop(cols_remove,axis=1)

	#remove special characters
	mdf=replace_special(mdf)

	#transform movements using pivot. The colums are the reason to move, the index are the global ids...
	#...and the values is the number of times the employee has performed that movement
	mdf = mdf.set_index('global id').reset_index()
	mdf = (mdf[['reason for action','global id']]
		.pivot_table(index = 'global id',columns='reason for action',aggfunc=len, fill_value=0)
		.reset_index())

	if idx_left:
		#initiate the label column
		mdf['label'] = 0
		
		#update the labels for the people not in the company
		mdf = mdf.set_index('global id')
		mdf.ix[idx_left,'label'] = 1
		mdf = mdf.reset_index()
		
	return mdf


#Add library files by Andre

def notebook_prepareDatasetTrainOpr(adls):

	#read information about new dataset
	df = pd.read_csv(adls.open(path_navigate+'opr.csv'))
	#preprocess dataset using defined script
	opr = webService_preProcessOprDataset(df)
	#add label to the dataset
	opr = webService_addLabel(opr,turnOverNext)
	#convert label to -1 +1 format
	opr['label'] = [1 if k==1 else -1 for k in opr['label']]

	save_csv_adls(opr,path_datasets+'processedOpr.csv',adls)

	return "process done" #return render(request,'finish_ws.html',context={'msg':'Opr dataset saved in Azure Data Lake'})
	
def notebook_prepareDatasetTrainMovements(adls):

	#read movements information for 2015
	moveCurrent = pd.read_csv(adls.open(path_head+'Movements_2015_Year_End.csv',),low_memory=False,engine='python')

	#convert the id to the standard format applied to all ids
	moveCurrent['global id'] = convert_id(moveCurrent['Pers.No.'])

	#get the set that contains all ids
	moveKeyCurrent = set(moveCurrent['global id'])
		
	#get the ids for the people who were not in the company
	idxCurrent=list(moveKeyCurrent.intersection(turnOverNext))    

	#use procedure to process raw dataset
	moveCurrent  = movements_procedure(moveCurrent,idxCurrent)

	moveCurrent.columns = ['movement --- '+k if k!='global id' and k!='label' else k for k in moveCurrent.columns]
	#moveCurrent.to_csv(path_datasets+'movements15.csv',index=False)
	save_csv_adls(moveCurrent,path_datasets+'processedMoveCurrent.csv',adls)

	return "process done" #return HttpResponse("Movements dataset saved in ADL.")

def notebook_prepareDatasetTrainCompetencies(adls):
	
	df = pd.read_csv(adls.open(path_navigate+'competencies.csv'),low_memory=False,engine='python')
	df['Global ID'] = convert_id(df['Employee Global ID'])
	#df=df[~df.duplicated('Global ID')]
	df.columns=replace_special(list(df.columns))
	cols_keep = ['global id','plan name','competency group','competency','manager rating - numeric value','competency reference key']
	df=replace_special(df[cols_keep])

	#df['competency'] = df['plan name'] + [' - '] + df['competency'] + [' - '] + df['competency reference key']
	#df = df.drop(['competency reference key','plan name'],axis=1)

	m = np.mean([float(k) for k in df['manager rating - numeric value'] if k!='not rated'])
	df['manager rating - numeric value']  = [float(k) if k!='not rated' else m for k in df['manager rating - numeric value']]

	aux1=df.groupby(['global id','plan name','competency']).agg({'manager rating - numeric value':'mean'}).reset_index()
	aux1['competency'] = aux1['plan name'] + [' - '] + aux1['competency']
	aux1 = aux1.drop(['plan name'],axis=1)
	aux1 = aux1.pivot(index='global id',columns='competency',values='manager rating - numeric value')

	gaux1 = get_stats(aux1)
	gaux1_col = list(gaux1[gaux1['not_null_pct']>.5]['col'].values)
	aux1 = aux1[gaux1_col].reset_index()

	aux2=df.groupby(['global id','competency group']).agg({'manager rating - numeric value':'mean'}).reset_index()
	aux2 = aux2.pivot(index='global id',columns='competency group',values='manager rating - numeric value')

	df=aux1.join(aux2,on='global id').fillna('empty')

	for col in list(set(df.columns) - set(['global id','label'])):
		m = np.mean([float(k) for k in df[col] if k!='empty'])
		df[col]  = [float(k) if k!='empty' else m for k in df[col]]
		df[col] = get_stds(df[col])


	dfk = set(df['global id'])
	#initiate the label column
	df['label'] = 0
	idxCurrent=list(dfk.intersection(turnOverNext))
	#update the labels for the people not in the company
	df = df.set_index('global id')
	df.ix[idxCurrent,'label'] = 1
	df = df.reset_index()
	#print df['label'].value_counts()
	#print df['label'].value_counts()/len(df)

	competencies_raw = df.copy(deep=True)
	df.columns = ['competencies --- '+k if k!='global id' and k!='label' else k for k in df.columns]
	competencies = df.copy(deep=True)
	save_csv_adls(competencies,path_datasets+'processedCompetencies.csv',adls)

	return "process done" #return HttpResponse("Competencies Dataset saved in ADL.")

def notebook_prepareDatasetTrainBandRatioChange(adls):

	df = pd.read_csv(adls.open(path_head+'movements_white_executives.csv'),low_memory=False,engine='python')
	df['Global ID'] = convert_id(df['Global ID'])
	df.columns=replace_special(list(df.columns))
	df.columns = [k.strip() for k in df.columns]

	cols_keep = ['global id','original hiring date','start date','changed on','end date','pay scale group']
	df=df[cols_keep]

	df['pay scale group'] = [k.strip().lower().replace(' ','-').replace('--','-').replace('us-','') for k in df['pay scale group']]
	df['psg converted'] = [band_dict[k] for k in df['pay scale group']]

	df['band delta'] = df['psg converted'].diff()
	idx=df[~df.duplicated('global id',keep='first')].index
	df.ix[idx,'band delta']=0

	l=[]
	for val in df['band delta']:
		if val<-2:
			a='bigger than -2'
		elif val<-1:
			a='between -2 and -1'
		elif val<0:
			a='between -1 and 0'
		elif val<1:
			a='between 0 and 1'
		elif val<=2:
			a='between 1 and 2'
		elif val>2:
			a='bigger than 2'
		l.append(a)
	df['band delta discrete'] = l

	p = df.set_index('global id').reset_index()
	p = (p[['band delta discrete','global id']]
		.pivot_table(index = 'global id',columns='band delta discrete',aggfunc=len, fill_value=0)
		.reset_index())
	p['band delta count'] = p.sum(axis=1)
	for col in p.drop(['global id'],axis=1).columns:
		p[col] = get_stds(p[col])
	band = p.copy(deep=True)

	bandk = set(band['global id'])

	#initiate the label column
	band['label'] = 0
	#get the ids for the people who were not in the company
	idxCurrent=list(bandk.intersection(turnOverNext))
	#update the labels for the people not in the company
	band = band.set_index('global id')
	band.ix[idxCurrent,'label'] = 1
	band = band.reset_index()

	band.columns = ['band ratio change --- '+k if k!='global id' and k!='label' else k for k in band.columns]
	#print band['label'].value_counts()
	#print band['label'].value_counts()/len(band)
	#band.to_csv(path_datasets+'band_ratio.csv',index=False)

	save_csv_adls(band,path_datasets+'processedBandRatioChange.csv',adls)

	return "process done" #return HttpResponse("Band Ratio Change dataset saved in ADL.")

def notebook_prepareDatasetTrainTimeInPosition(adls):
	df = pd.read_csv(adls.open(path_head+'movements_white_executives.csv'),low_memory=False,engine='python')
	df['Global ID'] = convert_id(df['Global ID'])
	df.columns=replace_special(list(df.columns))
	df.columns = [k.strip() for k in df.columns]

	cols_keep = ['global id','original hiring date','start date','changed on','end date','pay scale group']
	df=df[cols_keep]

	for col in ['original hiring date','end date','start date','changed on']:
		df[col] = [convert_dt(k,date_format="%d.%m.%Y") if type(k) is str else None for k in df[col]]
	df = df.sort_values(by=['global id','changed on'],ascending=True).reset_index(drop=True)
	df = df.replace(dt.datetime(9999, 12, 31, 0, 0),dt.datetime(2017, 12, 31, 0, 0))

	aux = df[['start date','end date','global id']].copy(deep=True)
	l = aux[['start date','end date']].apply(monthdelta_apply,axis=1)
	aux['time in position'] = l

	time = aux.groupby('global id').agg({'time in position':['mean','sum','std','max','min']})
	aux = aux[~aux.duplicated('global id',keep='last')].rename(columns={'time in position':'time in last position'})
	time = time.reset_index()
	time = time.join(aux[['global id','time in last position']].set_index('global id'),on='global id')
	time.columns = [str(k).replace('(','').replace(')','').replace(',',' - ').replace("'",'') for k in time.columns]
	time = time.rename(columns={'global id -  ':'global id'})

	st = list(df['start date'][1:].values)
	st = st + [None]
	df['return date'] = st
	aux = df[['return date','end date','global id']].copy(deep=True)
	l = aux[['end date','return date']].apply(monthdelta_apply,axis=1)
	aux['time away'] = l

	aux['count time away'] = (aux['time away']>0).astype(float)
	cta = aux.groupby('global id').agg({'count time away':'sum'})
	aux=aux[aux['time away']!=0]
	aux = aux.groupby('global id').agg({'time away':['mean','sum','std','max','min']})
	aux = aux.reset_index().join(cta,on='global id')
	aux.columns = [str(k).replace('(','').replace(')','').replace(',',' - ').replace("'",'') for k in aux.columns]
	aux = aux.rename(columns={'global id -  ':'global id'})
	time = time.join(aux.set_index('global id'),on='global id',how='left').fillna(0)

	timek = set(time['global id'])

	#initiate the label column
	time['label'] = 0
	#get the ids for the people who were not in the company
	idxCurrent=list(timek.intersection(turnOverNext))
	#update the labels for the people not in the company
	time = time.set_index('global id')
	time.ix[idxCurrent,'label'] = 1
	time = time.reset_index()

	time.columns = replace_special(list(time.columns))
	for num in [1,2,3,5,10,20]:
		time[str(num) + ' - position anniversary'] = (time['time in last position']>=num).astype(float)

	for col in time.drop(['label','global id'],axis=1).columns:
		time[col] = get_stds(time[col])

	time.columns = ['time in position --- '+k if k!='global id' and k!='label' else k for k in time.columns]

	save_csv_adls(time,path_datasets+'processedTimeInPosition.csv',adls)

	return "process done" #return HttpResponse("Time in Position dataset saved in ADL.")

def notebook_prepareDatasetTrainSalaryBonus(adls):

	df = pd.read_csv(adls.open(path_bonus+'salary_bonus.csv'),low_memory=False,dtype=str,engine='python')
	df['Global ID'] = convert_id(df['SAPID'])
	cols_remove = ['SAPID', 'global id',
				   'Company Bonus (Earned prior Yr)  Wt=1200',
				   'Company Bonus (Earned Prior Yr) WT=9A50',
				   'Incentive Bonuses for 2014,2015,2016  (WT=1230,1231,1240,1241,1245,1247)',
				   'Incentive Bonuses for Jan-2017', 'Incentive Bonus Feb-2017',
				   'Incentive Bonus Mar-2017', 'Incentive Bonus Apr-2017',
				   'Incentive Bonus May-2017']
	df=df.drop(cols_remove,axis=1)

	df=replace_special(df)
	cols = list(set(df.columns) - set(['year','global id']))
	df = df[(df['year']!='2016') & (df['year']!='2017')]

	ndf = pd.DataFrame(df['global id'].unique(),columns=['global id'])
	for col in cols:
		aux=df.pivot(columns='year',values=col,index='global id')
		ndf=ndf.join(aux,on='global id',rsuffix=' '+col)
	ndf=ndf.dropna(axis=0,how='any').reset_index(drop=True)
	ndf.columns = ['global id','2014 bonus total','2015 bonus total'] + list(ndf.columns.values)[3:]
	df=ndf

	#convert currency
	cols = ['2014 bonus total','2015 bonus total',
			'2014 annual salary', '2015 annual salary',
			'2014 grand total','2015 grand total']
	for col in cols:
		try:
			df[col] = df[col].replace( '[\$,)]','', regex=True ).replace( '[(]','-',   regex=True ).astype(float)
		except:
			pass
		
	df=df.dropna(how='any',axis=0)
	cols = ['2014 annual salary','2015 annual salary', '2014 grand total', '2015 grand total',
			'2014 bonus total', '2015 bonus total']
	for col in cols:
		df[col] = df[col].astype(float)

	df['2014 bonus salary pct'] = df['2014 bonus total']/df['2014 annual salary'].astype(float)
	df['2015 bonus salary pct'] = df['2015 bonus total']/df['2015 annual salary'].astype(float)

	c14 = ['2014 annual salary','2014 grand total','2014 bonus total','2014 bonus salary pct']
	c15 = ['2015 annual salary','2015 grand total','2015 bonus total','2015 bonus salary pct']

	for i in range(len(c14)):
		n = c14[i]
		n=n[n.index(' ')+1:]
		df['delta '+n] = (df[c15[i]] - df[c14[i]]).values
		if n in ['annual salary','grand total', 'bonus total']:
			df['delta '+n] = [k/df[c14[i]].iloc[j] if df[c14[i]].iloc[j]!=0 else k for j,k in enumerate(df['delta '+n])]

	cols = ['2014 band','2015 band']*3
	cols_agg = ['2014 annual salary','2015 annual salary','2014 grand total','2015 grand total',
				'2014 bonus total','2015 bonus total']
	df=get_aggregates(df,cols,cols_agg)

	df['2014 band'] = [float(roman2latin(k)) for k in df['2014 band']]
	df['2015 band'] = [float(roman2latin(k)) for k in df['2015 band']]

	cols = ['2014 bonus total', '2015 bonus total','2014 annual salary', 
			'2015 annual salary','2014 grand total', '2015 grand total',
			'2014 bonus salary pct','2015 bonus salary pct']
	for col in cols:
		df[col] = get_stds(df[col])
		
	df=df.replace([np.nan,np.inf],[None,None])
	deltas = [k for k in df.columns if 'delta' in k]
	for col in deltas:
		m=df[col].mean()
		l = get_stds(df[col])
		for i,v in enumerate(df[col]):
			if v is None: 
				l[i]=None
				print v
			elif v>=0: l[i] = l[i]+' positive'
			else: l[i] = l[i]+' negative'
		df[col] = l
		
	df.columns = [k.replace('2014','past').replace('2015','current') for k in df.columns]
		
	salbon=df.copy(deep=True)
	#initiate the label column
	salbon['label'] = 0
	#get the ids for the people who were not in the company
	salbonk = set(salbon['global id'])
	idxCurrent=list(salbonk.intersection(turnOverNext))
	#update the labels for the people not in the company
	salbon = salbon.set_index('global id')
	salbon.ix[idxCurrent,'label'] = 1
	salbon = salbon.reset_index()

	salbon.columns = replace_special(list(salbon.columns))
	salbon.columns = ['salbon --- '+k if k!='global id' and k!='label' else k for k in salbon.columns]
	save_csv_adls(salbon,path_datasets+'processedSalaryBonus.csv',adls)

	return "process done" #return HttpResponse("Salary & Bonus dataset saved in ADL.")

def notebook_prepareDatasetTrainTargetAchievement(adls):

	#read dataset for target achievement in the first semester
	df = pd.read_csv(adls.open(path_target+'2015_h1_average_payout.csv'),low_memory=False,engine='python')

	#prepare dataset
	cols_keep = ['Time Dedication Rate','Appraisee ID','Functional Area','Appraiser ID','Closest Entity SOP M','Upper Entity SOP M',
				'Band','Macro Entity','SOP','Individual Target','Entity Target','Bonus% by Band']
	df=df[cols_keep]

	df=replace_special(df)
	df=df.rename(columns={'appraisee id':'global id'})
	df['global id'] = convert_id(df['global id'])
	df['appraiser id'] = convert_id(df['appraiser id'])

	cols = ['time dedication rate','individual target','entity target','bonus% by band']
	for col in cols:
		df[col] = [float(k[:k.index('%')]) for k in df[col]]

	#remove appraiser id to test - reason: way too many individual values 888
	df['band'] = [float(roman2latin(k)) for k in df['band']]
	df=df[['global id','functional area',
		 'macro entity','sop','band','time dedication rate','individual target','entity target']]
	df=df.dropna(how='any',axis=0)
	values = (df['individual target']/df['entity target'].astype(float)).replace(np.inf,None)
	values = values.fillna(values.mean()+3*values.std())

	l=[]
	for v in values:
		if v<1: k='performance below entity'
		elif v==1: k='peformance equal entity'
		elif v<2: k='performance above entity'
		elif v<3: k='performance above twice entity'
		else: k= 'performance above tree times entity'
		l.append(k)

	df['individual target over entity added'] = l

	cols = ['band','sop','functional area','macro entity']
	cols_agg = ['individual target']*4
	df=get_aggregates(df,cols,cols_agg)
	df=df.dropna(how='any',axis=0)

	targetCurrentFirstSemester = df.copy(deep=True)
	#initiate the label column
	targetCurrentFirstSemester['label'] = 0
	#get the ids for the people who were not in the company
	targetCurrentFirstSemesterKey = set(targetCurrentFirstSemester['global id'])
	idxCurrent=list(targetCurrentFirstSemesterKey.intersection(turnOverNext))
	#update the labels for the people not in the company
	targetCurrentFirstSemester = targetCurrentFirstSemester.set_index('global id')
	targetCurrentFirstSemester.ix[idxCurrent,'label'] = 1
	targetCurrentFirstSemester = targetCurrentFirstSemester.reset_index()
	#targetCurrentFirstSemester.to_csv(path_datasets+'targetCurrentFirstSemester.csv',index=False)

	#read dataset for target achievement in the second semester
	df = pd.read_csv(adls.open(path_target+'2015_h2_average_payout.csv'),low_memory=False,engine='python')

	#prepare dataset
	cols_keep = ['Time Dedication Rate','Appraisee ID','Functional Area','Appraiser ID','Closest Entity SOP M','Upper Entity SOP M',
				'Band','Macro Entity','SOP','Individual Target','Entity Target','Bonus% by Band']
	df=df[cols_keep]

	df=replace_special(df)
	df=df.rename(columns={'appraisee id':'global id'})
	df['global id'] = convert_id(df['global id'])
	df['appraiser id'] = convert_id(df['appraiser id'])
	df = df.dropna(how='any')

	cols = ['time dedication rate','individual target','entity target','bonus% by band']
	for col in cols:
		df[col] = [float(k[:k.index('%')]) for k in df[col]]

	#remove appraiser id to test - reason: way too many individual values 888
	df['band'] = [float(roman2latin(k)) for k in df['band']]
	df=df[['global id','functional area',
		 'macro entity','sop','band','time dedication rate','individual target','entity target']]
	df=df.dropna(how='any',axis=0)
	values = (df['individual target']/df['entity target'].astype(float)).replace(np.inf,None)
	values = values.fillna(values.mean()+3*values.std())

	l=[]
	for v in values:
		if v<1: k='performance below entity'
		elif v==1: k='peformance equal entity'
		elif v<2: k='performance above entity'
		elif v<3: k='performance above twice entity'
		else: k= 'performance above tree times entity'
		l.append(k)

	df['individual target over entity added'] = l

	cols = ['band','sop','functional area','macro entity']
	cols_agg = ['individual target']*4
	df=get_aggregates(df,cols,cols_agg)
	df=df.dropna(how='any',axis=0)

	targetCurrentSecondSemester = df.copy(deep=True)
	#initiate the label column
	targetCurrentSecondSemester['label'] = 0
	#get the ids for the people who were not in the company
	targetCurrentSecondSemesterKey = set(targetCurrentSecondSemester['global id'])
	idxCurrent=list(targetCurrentSecondSemesterKey.intersection(turnOverNext))
	#update the labels for the people not in the company
	targetCurrentSecondSemester = targetCurrentSecondSemester.set_index('global id')
	targetCurrentSecondSemester.ix[idxCurrent,'label'] = 1
	targetCurrentSecondSemester = targetCurrentSecondSemester.reset_index()
	#targetCurrentSecondSemester.to_csv(path_datasets+'targetCurrentSecondSemester.csv',index=False)

	targetCurrentKey = list(targetCurrentSecondSemesterKey.intersection(targetCurrentFirstSemesterKey))
	targetCurrent = targetCurrentFirstSemester.set_index('global id').loc[targetCurrentKey].reset_index()
	cols2 = ['time dedication rate', 'individual target', 'entity target',
		   'individual target over entity added', 'individual target by band',
		   'individual target by sop', 'individual target by functional area',
		   'individual target by macro entity']
	t2=targetCurrentSecondSemester.set_index('global id').loc[targetCurrentKey][cols2]
	t2.columns = [k+' 2' for k in t2.columns]
	targetCurrent = targetCurrent.join(t2,on='global id',how='inner')

	targetCurrent.columns = replace_special(list(targetCurrent.columns))
	targetCurrent.columns = ['target --- '+k if k!='global id' and k!='label' else k for k in targetCurrent.columns]
	#targetCurrent.to_csv(path_datasets+'targetCurrent.csv',index=False)
	save_csv_adls(targetCurrent,path_datasets+'processedTargetAchievement.csv',adls)

	return "process done" #return HttpResponse("Target Achievement dataset saved in ADL.")

def notebook_prepareDatasetTrainHeadCount(adls):

	#read headcount information for 2014
	headPast = pd.read_csv(adls.open(path_head+'Headcount_2014_Year_End.csv'),low_memory=False,engine='python')

	#convert the id to the standard format applied to all ids
	headPast['Global ID'] = convert_id(headPast['Global ID'])

	#get the set that contains all ids
	headKeyPast = set(headPast['Global ID'])

	#read headcount information for 2015
	headCurrent = pd.read_csv(adls.open(path_head+'Headcount_2015_Year_End.csv'),low_memory=False)

	#convert the id to the standard format applied to all ids
	headCurrent['Global ID'] = convert_id(headCurrent['Global ID'])

	#get the set that contains all ids
	headKeyCurrent = set(headCurrent['Global ID'])

	#get the ids for the people who were not in the company in 2014
	idxPast=list(headKeyPast.intersection(turnOverCurrent))

	#use procedure to process raw dataset
	headPast = headcount_procedure(headPast,idxPast)

	#print information about the distribuitions of the labels in the dataset
	#print headPast['label'].value_counts()
	#print headPast['label'].value_counts()/len(headPast)

	#create a version with the simple name columns
	headPast_raw = headPast.copy(deep=True)

	#add information about the dataset in the name of the column in order to identify the source
	headPast.columns = ['head --- '+k if k!= 'global id' and k!='label' else k for k in headPast.columns ]

	#remove special characters
	headPast = replace_special(headPast)

	#save the dataset created
	#headPast.drop(['head --- personnel number manager'],axis=1).to_csv(path_datasets+'head14.csv',index=False)
	#save_csv_adls(headPast.drop(['head --- personnel number manager'],axis=1),path_datasets+'head14.csv',adls)
	headPast = headPast.drop(['head --- personnel number manager'],axis=1)

	#get the ids for the people who were not in the company in 2015
	idxCurrent=list(headKeyCurrent.intersection(turnOverNext))

	#use procedure to process raw dataset
	headCurrent = headcount_procedure(headCurrent,idxCurrent)

	#add information about manager changing or leaving between 2014 and 2015
	##slice the information from headcount 14
	aux1 = headPast_raw[['personnel number manager','label','global id']]

	##slice the information from headcount 15
	aux2 = headCurrent[['personnel number manager','label','global id']]

	##join the sliced dataframes in order to compare the columns
	aux = aux2.join(aux1.set_index('global id'),on='global id',how='left',rsuffix='_14')

	##calculate if the manager global id is difeerent in the two years
	aux['manager change'] = aux['personnel number manager'] != aux['personnel number manager_14']

	##calculate if the manager has left in 2014
	aux['manager left'] = aux['label_14']

	##keep only the two columns calculated
	aux = aux[['global id','manager change','manager left']].fillna(0)

	##merge the information calculated with the original dataset
	headCurrent = headCurrent.join(aux.set_index('global id'),on='global id')

	#print information about the distribuitions of the labels in the dataset
	#print headCurrent['label'].value_counts()
	#print '----'
	#print headCurrent['label'].value_counts()/len(headCurrent)

	#add information about the dataset in the name of the column in order to identify the source
	headCurrent.columns = ['head --- '+k if k!= 'global id' and k!='label' else k for k in headCurrent.columns]

	#remove special characters
	headCurrent = replace_special(headCurrent)

	#save the dataset created
	#headCurrent.drop(['head --- personnel number manager'],axis=1).to_csv(path_datasets+'head15.csv',index=False)
	save_csv_adls(headCurrent.drop(['head --- personnel number manager'],axis=1),path_datasets+'processedHeadCurrent.csv',adls)
	
	return "process done" #return HttpResponse("Headcount dataset saved in ADL.")

def analysis_procedure(df1,df2=None,method='intra',categorical=None,numerical=None,
					   return_all=False,sm=None,label_name='label',id_name='global id'):
	if method =='intra':
		df = df1.reset_index(drop=True)
	elif method == 'inter':
		df=df1.append(df2,ignore_index=True).reset_index(drop=True)
	else:
		print 'method passed is not valid. Options "intra" and "inter"'
		return None
		
	X = df.drop([label_name,id_name],axis=1)
	y = df[label_name].astype(float).values
	oh = None
	lle= None
	
	if categorical is not None:
		x_categorical = df[categorical]
		Xoh,oh,Xle,lle = get_dummies_sl(x_categorical,return_all=True)
		x_categorical = pd.DataFrame(Xoh.toarray()).reset_index(drop=True)
	if numerical is not None:    
		x_numerical = df[numerical].reset_index(drop=True)
	if categorical is not None and numerical is not None:
		X = pd.concat([x_categorical,x_numerical],ignore_index=True,axis=1)
	elif numerical is not None:
		X = x_numerical
	elif categorical is not None:
		X = x_categorical
		
	if sm:
		Xs,ys = sm.fit_sample(X,y)
		xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=.3,random_state=1,stratify=y)
		xstrain,xstest,ystrain,ystest = train_test_split(Xs,ys,test_size=.3,random_state=1,stratify=ys)
		xtrain = xstrain
		ytrain = ystrain
	elif method=='intra':
		xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)
	elif method=='inter':
		l1=len(df1)
		l2=len(df2)
		xtrain = X.loc[range(0,l1)]
		xtest = X.loc[range(l1,l1+l2)]
		ytrain = df1[label_name].astype(float).values
		ytest = df2[label_name].astype(float).values
	else:
		print 'method passed is not valid. Options "intra" and "inter"'
		return None
	
	clf=LogisticRegression()
	print cross_val_score(clf,X,y,cv=3,scoring='f1')
	if sm:
		print cross_val_score(clf,Xs,ys,cv=3,scoring='f1')
		clf.fit(Xs,ys)
	else:
		clf.fit(X,y)
	r=pd.DataFrame(get_pred_results(clf,xtrain,ytrain,xtest,ytest))
	if return_all:
		return r,clf,oh,lle,X,y
	return r

def get_pred_results(clf,X_train,y_train,X_test,y_test,model='model',verbose=False,plot=False,class_focus=1):
	#fit the model
	clf.fit(X_train,y_train)

	#get prediction results
	pred = clf.predict(X_test)
	prob = clf.predict_proba(X_test)[:,1]
	
	a = [accuracy_score(y_test,pred)]
	p = [k[class_focus] for k in precision_recall_fscore_support(y_test,pred)]
	r = [[model]+a+p]
	
	if verbose:
		print model
		print a
		print p
		print '---'

def get_weights(clf,lle=None,numerical=[],func_apply='mean'):
	l1=[]
	l2=[]
	if lle:
		for k1 in lle.keys():
			for k2 in lle[k1].keys():
				l1.append(k1)
				l2.append(k2)
	l1+=numerical
	l2+=numerical
	
	gen=pd.DataFrame([l1,l2,list(clf.coef_[0])]).T
	gen.columns = ['variable','value','weight']
	gen['weight']=gen['weight'].astype(float)
	gen['dataset'] = [k[:k.index(' --- ')] if ' --- ' in k else k for k in gen['variable']]
	gen['variable'] = [k[k.index(' --- ')+5:] if ' --- ' in k else k for k in gen['variable']]
	gen['value'] = [k[k.index(' --- ')+5:] if ' --- ' in str(k) else k for k in gen['value']]
	
	avg_variable = gen.groupby('variable').agg({'weight':func_apply})
	avg_variable=avg_variable.reset_index()
	
	avg_dataset = gen.groupby('dataset').agg({'weight':func_apply})
	avg_dataset= avg_dataset.reset_index()
	return gen,avg_variable,avg_dataset

def band_change_procedure(df,cap_year,idx_left=None):
	df.columns=replace_special(list(df.columns))
	df.columns = [k.strip() for k in df.columns]

	cols_keep = ['global id','original hiring date','start date','changed on','end date','pay scale group']
	df=df[cols_keep]
    
	change_year = df['changed on'].apply(lambda x: datetime.strptime(x,"%d.%m.%Y").year)
	change_year = change_year<=cap_year
	df=df[change_year]

	df['pay scale group'] = [k.strip().lower().replace(' ','-').replace('--','-').replace('us-','') for k in df['pay scale group']]
	df['psg converted'] = [band_dict[k] for k in df['pay scale group']]

	df['band delta'] = df['psg converted'].diff()
	idx=df[~df.duplicated('global id',keep='first')].index
	df.ix[idx,'band delta']=0

	l=[]
	for val in df['band delta']:
		if val<-2:
			a='bigger than -2'
		elif val<-1:
			a='between -2 and -1'
		elif val<0:
			a='between -1 and 0'
		elif val<1:
			a='between 0 and 1'
		elif val<=2:
			a='between 1 and 2'
		elif val>2:
			a='bigger than 2'
		l.append(a)
	df['band delta discrete'] = l

	p = df.set_index('global id').reset_index()
	p = (p[['band delta discrete','global id']]
		.pivot_table(index = 'global id',columns='band delta discrete',aggfunc=len, fill_value=0)
		.reset_index())
	p['band delta count'] = p.sum(axis=1)
	for col in p.drop(['global id'],axis=1).columns:
		p[col] = get_stds(p[col])
	band = p.copy(deep=True)

	
	if idx_left:
		#initiate the label column
		band['label'] = 0
		#update the labels for the people not in the company
		idx_left=set(idx_left).intersection(set(band['global id']))
		band = band.set_index('global id')
		band.ix[idx_left,'label'] = 1
		band = band.reset_index()
	
	return band

def time_procedure(df,cap_year,idx_left=None):
	df.columns=replace_special(list(df.columns))
	df.columns = [k.strip() for k in df.columns]

	cols_keep = ['global id','original hiring date','start date','changed on','end date','pay scale group']
	df=df[cols_keep]
    
	change_year = df['changed on'].apply(lambda x: datetime.strptime(x,"%d.%m.%Y").year)
	change_year = change_year<=cap_year
	df=df[change_year]

	for col in ['original hiring date','end date','start date','changed on']:
		df[col] = [convert_dt(k,date_format="%d.%m.%Y") if type(k) is str else None for k in df[col]]
	df = df.sort_values(by=['global id','changed on'],ascending=True).reset_index(drop=True)
	df = df.replace(dt.datetime(9999, 12, 31, 0, 0),dt.datetime(2017, 12, 31, 0, 0))

	aux = df[['start date','end date','global id']].copy(deep=True)
	l = aux[['start date','end date']].apply(monthdelta_apply,axis=1)
	aux['time in position'] = l

	time = aux.groupby('global id').agg({'time in position':['mean','sum','std','max','min']})
	aux = aux[~aux.duplicated('global id',keep='last')].rename(columns={'time in position':'time in last position'})
	time = time.reset_index()
	time = time.join(aux[['global id','time in last position']].set_index('global id'),on='global id')
	time.columns = [str(k).replace('(','').replace(')','').replace(',',' - ').replace("'",'') for k in time.columns]
	time = time.rename(columns={'global id -  ':'global id'})

	st = list(df['start date'][1:].values)
	st = st + [None]
	df['return date'] = st
	aux = df[['return date','end date','global id']].copy(deep=True)
	l = aux[['end date','return date']].apply(monthdelta_apply,axis=1)
	aux['time away'] = l

	aux['count time away'] = (aux['time away']>0).astype(float)
	cta = aux.groupby('global id').agg({'count time away':'sum'})
	aux=aux[aux['time away']!=0]
	aux = aux.groupby('global id').agg({'time away':['mean','sum','std','max','min']})
	aux = aux.reset_index().join(cta,on='global id')
	aux.columns = [str(k).replace('(','').replace(')','').replace(',',' - ').replace("'",'') for k in aux.columns]
	aux = aux.rename(columns={'global id -  ':'global id'})
	time = time.join(aux.set_index('global id'),on='global id',how='left').fillna(0)

	time.columns = replace_special(list(time.columns))
	for num in [1,2,3,5,10,20]:
		time[str(num) + ' - position anniversary'] = (time['time in last position']>=num).astype(float)

	for col in time.drop(['global id'],axis=1).columns:
		time[col] = get_stds(time[col])
		
	if idx_left:
		#initiate the label column
		time['label'] = 0
		#get the ids for the people who were not in the company
		idx_left=set(idx_left).intersection(set(time['global id']))
		#update the labels for the people not in the company
		time = time.set_index('global id')
		time.ix[idx_left,'label'] = 1
		time = time.reset_index()
	return time

def opr_procedure(df,idx_left=None,max_year=None,min_year=None):
	cols_remove = ['Personnel Number', 'First Name', 'Last Name','Zone Name',
				   'Final Evaluation Workflow State','Review Period','Employee Global ID']
	df=df.drop(cols_remove,axis=1)

	df=df[(df['Year']<=max_year) & (df['Year']>=min_year)]
	unique_year = df['Year'].unique()
	df['Year'] = df['Year'].replace(unique_year,range(len(unique_year)))


	#special modifications for opr
	f=(df['OPR Rating Scale']=='1A') | (df['OPR Rating Scale']=='1B')
	#df=df[~f].reset_index(drop=True)
	p=df[['Year','OPR Rating Scale','Global ID']]
	p=p[~p[['Global ID','Year']].duplicated()]
	p=p.pivot(index='Global ID',columns='Year',values='OPR Rating Scale').reset_index().set_index('Global ID')
	p=p[[k for k in p.columns if k!='Year']]
	p.columns = ['opr '+str(k) for k in p.columns]

	df=df[~df.duplicated('Global ID')].drop(['Year','OPR Rating Scale','Plan Name'],axis=1)
	df=df.join(p,on='Global ID',how='left')
	opr_col=[k for k in df.columns if 'opr' in str(k)]
	opr=df.copy(deep=True)
	l=[]
	for k in opr['Position Title']:
		if 'dir' in str(k).lower():
			l.append('director')
		elif 'manager' in str(k).lower() or 'mgr' in str(k).lower() or 'gm,' in str(k).lower():
			l.append('manager')
		elif 'spec' in str(k).lower():
			l.append('specialist')
		elif 'engineer' in str(k).lower():
			l.append('engineer')
		elif 'vp,' in str(k).lower() or 'vp.' in str(k).lower() or 'vice president' in str(k).lower() or 'vp ' in str(k).lower():
			l.append('vice president')
		elif 'analyst' in str(k).lower():
			l.append('analyst')
		elif 'coordinator' in str(k).lower():
			l.append('coordinator')
		elif 'supervisor' in str(k).lower():
			l.append('supervisor')
		elif 'sales' in str(k).lower() or 'commercial' in str(k).lower():
			l.append('comercial')
		elif 'admin' in str(k).lower() or 'planner' in str(k).lower():
			l.append('administrative')
		elif 'brewmaster' in str(k).lower():
			l.append('brewmaster')
		elif 'maint' in str(k).lower():
			l.append('maintenance')
		elif 'associate' in str(k).lower():
			l.append('associate')
		elif 'temporary' in str(k).lower():
			l.append('temporary')
		elif 'controller' in str(k).lower():
			l.append('controller')
		elif 'merch' in str(k).lower():
			l.append('merchand')
		elif 'tech' in str(k).lower():
			l.append('technician')
		elif 'bpm' in str(k).lower():
			l.append('bpm')
		elif 'legal' in str(k).lower() or 'counsel' in str(k).lower():
			l.append('legal')
		elif 'fsem' in str(k).lower():
			l.append('fsem')
		elif 'accountant' in str(k).lower():
			l.append('accountant')
		elif 'assoc' in str(k).lower():
			l.append('associate')
		elif 'chef' in str(k).lower():
			l.append('chef')
		elif 'coord' in str(k).lower():
			l.append('coordinator')
		elif 'supv' in str(k).lower():
			l.append('supervisor')
		elif 'lead' in str(k).lower():
			l.append('lead')
		elif 'rep.' in str(k).lower():
			l.append('rep')
		elif 'president' in str(k).lower():
			l.append('president')
		elif 'head' in str(k).lower():
			l.append('head')
		elif 'architect' in str(k).lower():
			l.append('architect')
		elif 'ppm' in str(k).lower():
			l.append('ppm')
		elif 'personnel' in str(k).lower():
			l.append('personnel')
		elif 'no name available' in str(k).lower():
			l.append('not available')
		elif 'attendant' in str(k).lower():
			l.append('attendant')
		elif 'driver' in str(k).lower():
			l.append('driver')
		elif 'educa' in str(k).lower():
			l.append('education')
		elif 'intern' in str(k).lower():
			l.append('intern')
		elif 'foreman' in str(k).lower():
			l.append('foreman')
		elif 'sme' in str(k).lower():
			l.append('sme')
		elif 'spm' in str(k).lower():
			l.append('spm')  
		else:
			l.append(str(k))
	df['Position Title'] = l

	df=df.replace(['1A','1B','3A','3B','4A','4B','Not Rated','2'],[1.5,1.0,3.5,3,4.5,4,None,None])
	opr_ori = df[opr_col].copy(deep=True)
	#df=df[~opr_ori.isnull().all(axis=1)]

	opr=df.copy(deep=True)
	opr.columns = replace_special(list(opr.columns))
	opr_raw = opr.copy(deep=True)

	df['OPR mean abs'] = pd.cut(opr_ori.mean(axis=1),bins=[k/10.0 for k in range(10,60,5)]).astype(str)
	df['OPR max abs'] = pd.cut(opr_ori.max(axis=1),bins=[k/10.0 for k in range(10,60,5)]).astype(str)
	df['OPR min abs'] = pd.cut(opr_ori.min(axis=1),bins=[k/10.0 for k in range(10,60,5)]).astype(str)

	df['OPR mean'] = opr_ori.mean(axis=1)
	df['OPR max'] = opr_ori.max(axis=1)
	df['OPR min'] = opr_ori.min(axis=1)
	df['OPR std'] = opr_ori.std(axis=1)


	cols = ['Position Title']*4 +  ['Function Name Level 2']*4
	cols_agg = ['OPR mean','OPR min','OPR max','OPR std']*2
	df=get_aggregates(df,cols,cols_agg)

	for col in ['OPR mean','OPR min','OPR max','OPR std']:   
		df[col] = get_stds(df[col])

	opr=df.copy(deep=True)
	
	if idx_left:
		#initiate the label column
		opr['label'] = 0
		#update the labels for the people not in the company
		idx_left = list(set(idx_left).intersection(set(opr['Global ID'])))
		opr = opr.set_index('Global ID')
		opr.ix[idx_left,'label'] = 1
		opr = opr.reset_index()

	opr.columns = replace_special(list(opr.columns))
	return opr

def competencies_procedure(df,idx_left):

	df.columns=replace_special(list(df.columns))
	cols_keep = ['global id','plan name','competency group','competency','manager rating - numeric value','competency reference key']
	df=replace_special(df[cols_keep])

	#df['competency'] = df['plan name'] + [' - '] + df['competency'] + [' - '] + df['competency reference key']
	#df = df.drop(['competency reference key','plan name'],axis=1)

	df=df.replace('not applicable','not rated')
	m = np.mean([float(k) for k in df['manager rating - numeric value'] if k!='not rated' ])
	df['manager rating - numeric value']  = [float(k) if k!='not rated' else m for k in df['manager rating - numeric value']]

	#aux1=df.groupby(['global id','plan name','competency']).agg({'manager rating - numeric value':'mean'}).reset_index()
	#aux1['competency'] = aux1['plan name'] + [' - '] + aux1['competency']
	#aux1 = aux1.drop(['plan name'],axis=1)
	#aux1 = aux1.pivot(index='global id',columns='competency',values='manager rating - numeric value')

	#gaux1 = get_stats(aux1)
	#gaux1_col = list(gaux1[gaux1['not_null_pct']>.5]['col'].values)
	#aux1 = aux1[gaux1_col].reset_index()
	#aux1=aux1.reset_index()

	aux2=df.groupby(['global id','competency group']).agg({'manager rating - numeric value':'mean'}).reset_index()
	aux2 = aux2.pivot(index='global id',columns='competency group',values='manager rating - numeric value')

	#df=aux1.join(aux2,on='global id').fillna('empty')
	df=aux2.reset_index()

	for col in list(set(df.columns) - set(['global id','label'])):
		m = np.mean([float(k) for k in df[col] if k!='empty'])
		df[col]  = [float(k) if k!='empty' else m for k in df[col]]
		df[col] = get_stds(df[col])

	if idx_left:
		#initiate the label column
		df['label'] = 0
		#update the labels for the people not in the company
		df = df.set_index('global id')
		df.ix[idx_left,'label'] = 1
		df = df.reset_index()

	competencies = df.copy(deep=True)
	
	return competencies

def opr_manager_procedure(headCurrent,opr_raw,label=True):
	aux=headCurrent[['global id','head --- personnel number manager']]
	aux.columns = ['global id','personnel number manager']
	aux = opr_raw.join(aux.set_index('global id'),on='global id',how='inner').reset_index(drop=True)

	l=[]
	gid = list(aux['global id'].astype(str))
	for k in aux['personnel number manager'].values:
		try:
			idx = gid.index(str(k))
			a=idx
		except Exception as e:
			a=None
		l.append(a)
	l=pd.Series(l)
	aux1=aux.ix[l]

	aux1.columns = ['manager - '+k for k in aux1.columns]
	aux1['global id'] = aux['global id'].values
	aux1 = aux1.set_index('global id').reset_index()
	aux1['manager - global id'] = aux1['manager - global id'].replace([np.nan],[None])
	df=aux1[~pd.isnull(aux1['manager - global id'])].reset_index(drop=True)
	df = df.join(opr_raw.set_index('global id'),on='global id',how='inner')

	if label:
		print df['label'].value_counts()
		print df['label'].value_counts()/len(df)
		df= df.drop(['manager - label','manager - global id','manager - personnel number manager'],axis=1)
	else:
		df= df.drop(['manager - global id','manager - personnel number manager'],axis=1)
		
	return df

def competencies_manager_procedure(headCurrent,competencies_raw,label=False):
	aux=headCurrent[['global id','head --- personnel number manager']]
	aux.columns = ['global id','personnel number manager']
	aux = competencies_raw.join(aux.set_index('global id'),on='global id',how='inner').reset_index(drop=True)

	l=[]
	gid = list(aux['global id'].astype(str))
	for k in aux['personnel number manager'].values:
		try:
			idx = gid.index(str(k))
			a=idx
		except Exception as e:
			a=None
		l.append(a)
	l=pd.Series(l)
	aux1=aux.ix[l]

	aux1.columns = ['manager - '+k for k in aux1.columns]
	aux1['global id'] = aux['global id'].values
	aux1 = aux1.set_index('global id').reset_index()
	aux1['manager - global id'] = aux1['manager - global id'].replace([np.nan],[None])
	df=aux1[~pd.isnull(aux1['manager - global id'])].reset_index(drop=True)
	df = df.join(competencies_raw.set_index('global id'),on='global id',how='inner')
	
	if label:
		print df['label'].value_counts()
		print df['label'].value_counts()/len(df)
		df = df.drop(['manager - label','manager - global id','manager - personnel number manager'],axis=1)
	else:
		df = df.drop(['manager - global id','manager - personnel number manager'],axis=1)
		
	return df

def salary_bonus_procedure(df,min_year,max_year,idx_left=None):
	cols_remove = ['SAPID', 'global id',
				   'Company Bonus (Earned prior Yr)  Wt=1200',
				   'Company Bonus (Earned Prior Yr) WT=9A50',
				   'Incentive Bonuses for 2014,2015,2016  (WT=1230,1231,1240,1241,1245,1247)',
				   'Incentive Bonuses for Jan-2017', 'Incentive Bonus Feb-2017',
				   'Incentive Bonus Mar-2017', 'Incentive Bonus Apr-2017',
				   'Incentive Bonus May-2017']
	df=df.drop(cols_remove,axis=1)

	df=replace_special(df)
	cols = list(set(df.columns) - set(['year','global id']))
	df['year']=df['year'].astype(int)
	df=df[(df['year']<=max_year) & (df['year']>=min_year) ]

	ndf = pd.DataFrame(df['global id'].unique(),columns=['global id'])
	for col in cols:
		aux=df.pivot(columns='year',values=col,index='global id')
		ndf=ndf.join(aux,on='global id',rsuffix=' '+col)
	ndf=ndf.dropna(axis=0,how='any').reset_index(drop=True)

	years=df['year'].unique()
	l=['global id']
	l+= [str(year)+' bonus total' for year in years]
	l+= [str(year)+' band' for year in years]
	l+= [str(year)+' annual salary' for year in years]
	l+= [str(year)+' grand total' for year in years]

	ndf.columns = l
	df=ndf
	#convert currency
	cols = [k for k in df.columns if ('band' not in k and 'global id' not in k)]
	for col in cols:
		try:
			df[col] = df[col].replace( '[\$,)]','', regex=True ).replace( '[(]','-',   regex=True ).astype(float)
		except:
			pass

	df=df.dropna(how='any',axis=0)
	for col in cols:
		df[col] = df[col].astype(float)

	for year in years:
		df[str(year)+' bonus salary pct'] = df[str(year)+' bonus total']/df[str(year)+' annual salary'].astype(float)

	past_year = str(years[-2:-1][0])
	past = [past_year+' annual salary',past_year+' grand total',
				past_year+' bonus total',past_year+' bonus salary pct']

	current_year = str(years[-1:][0])
	current = [current_year+' annual salary',current_year+' grand total',
				current_year+' bonus total',current_year+' bonus salary pct']

	for i in range(len(past)):
		n = past[i]
		n=n[n.index(' ')+1:]
		df['delta '+n] = (df[current[i]] - df[past[i]]).values
		if n in ['annual salary','grand total', 'bonus total']:
			df['delta '+n] = [k/df[past[i]].iloc[j] if df[past[i]].iloc[j]!=0 else k for j,k in enumerate(df['delta '+n])]

	cols = [k for k in df.columns if 'band' in k]
	cols_agg = [k for k in df.columns if ('band' not in k and 'global id' not in k)]
	df=get_aggregates(df,cols*3,cols_agg)
	for col in cols:
		df[col] = [float(roman2latin(k)) for k in df[col]]


	for year in years:
		cols = [str(year)+' bonus total',str(year)+' annual salary',str(year)+' grand total',str(year)+' bonus salary pct']
		for col in cols:
			df[col] = get_stds(df[col])

	df=df.replace([np.nan,np.inf],[None,None])
	deltas = [k for k in df.columns if 'delta' in k]
	for col in deltas:
		m=df[col].mean()
		l = get_stds(df[col])
		for i,v in enumerate(df[col]):
			if v is None: 
				l[i]=None
				print v
			elif v>=0: l[i] = l[i]+' positive'
			else: l[i] = l[i]+' negative'
		df[col] = l


	df.columns = [k.replace(past_year,'past').replace(current_year,'current') for k in df.columns]
	salbon=df.copy(deep=True)
	
	if idx_left:
		#initiate the label column
		salbon['label'] = 0
		#get the ids for the people who were not in the company
		#update the labels for the people not in the company
		salbon = salbon.set_index('global id')
		salbon.ix[idx_left,'label'] = 1
		salbon = salbon.reset_index()
		
	salbon.columns = replace_special(list(salbon.columns))

	return salbon

def target_procedure(df,idx_left=None):
	#prepare dataset
	cols_keep = ['Time Dedication Rate','Appraisee ID','Functional Area','Appraiser ID','Closest Entity SOP M','Upper Entity SOP M',
				'Band','Macro Entity','SOP','Individual Target','Entity Target','Bonus% by Band']
	df=df[cols_keep]

	df=replace_special(df)
	df=df.rename(columns={'appraisee id':'global id'})
	df['global id'] = convert_id(df['global id'])
	df['appraiser id'] = convert_id(df['appraiser id'])

	df['bonus% by band'] = df['bonus% by band'].fillna('20%')

	cols = ['time dedication rate','individual target','entity target','bonus% by band']
	for col in cols:
		df[col] = [float(k[:k.index('%')]) for k in df[col]]

	#remove appraiser id to test - reason: way too many individual values 888
	df['band'] = df['band'].fillna(np.nan)
	df['band'] = [k.strip().lower().replace(' ','-').replace('--','-').replace('us-','') if k is not np.nan else k for k in df['band']]

	###convert each value in pay scale group to a numerical band
	df['band'] = [band_dict[k] if k is not np.nan else k for k in df['band']]

	df=df[['global id','functional area',
		 'macro entity','sop','band','time dedication rate','individual target','entity target']]
	df=df.dropna(how='any',axis=0)
	values = (df['individual target']/df['entity target'].astype(float)).replace(np.inf,None)
	values = values.fillna(values.mean()+3*values.std())

	l=[]
	for v in values:
		if v<1: k='performance below entity'
		elif v==1: k='peformance equal entity'
		elif v<2: k='performance above entity'
		elif v<3: k='performance above twice entity'
		else: k= 'performance above tree times entity'
		l.append(k)

	df['individual target over entity added'] = l

	cols = ['band','sop','functional area','macro entity']
	cols_agg = ['individual target']*4
	df=get_aggregates(df,cols,cols_agg)
	df=df.dropna(how='any',axis=0)

	target = df.copy(deep=True)

	if idx_left:

		#initiate the label column
		target['label'] = 0
		#update the labels for the people not in the company
		target = target.set_index('global id')
		target.ix[idx_left,'label'] = 1
		target = target.reset_index()
		
	return target

def scale_pos_neg(df):
	l=[]
	oidx = df.index.values
	for col in df.columns:
		p=df[col][df[col]<0]
		n=df[col][df[col]>=0]
		if len(n)>0 and len(p)>0:
			min=-1
			max=1
		else:
			min=0
			max=1

		aux=df[col][df[col]<0]
		s=abs(aux.sum())
		#if s==0:
		#    print col
		#    print 'neg'
		for idx in aux.index:
			if s==0:
				aux[idx]=0
			else:
				aux[idx] = ((aux[idx]/s)-min)/(max-min)
		neg=aux.copy(deep=True)

		aux=df[col][df[col]>=0]
		s=aux.sum()
		#if s==0:
		#    print col
		#    print 'pos'
		for idx in aux.index:
			if s==0:
				aux[idx]=0
			else:
				aux[idx] = ((aux[idx]/s)-min)/(max-min)
		pos=aux.copy(deep=True)

		aux=pos.append(neg)
		aux=aux[oidx]
		s=aux.sum()
		for idx in aux.index:
			if s==0:
				aux[idx]=0
			else:
				aux[idx] = (aux[idx]/s)*100
		l.append(aux.values)
	df=pd.DataFrame(l,columns=oidx)
	return df

def pos_neg(df):
	l=[]
	oidx = df.index.values
	for col in df.columns:
		min=0
		max=1

		aux=df[col][df[col]<0]
		s=abs(aux.sum())
		for idx in aux.index:
			if s==0:
				aux[idx] = 0
			else:
				aux[idx] = ((aux[idx]/s)-min)/(max-min)*100
		neg=aux.copy(deep=True)

		aux=df[col][df[col]>=0]
		s=aux.sum()
		for idx in aux.index:
			if s==0:
				aux[idx] = 0
			else:
				aux[idx] = ((aux[idx]/s)-min)/(max-min)*100
		pos=aux.copy(deep=True)
		l.append(pos.append(neg)[oidx])
	return pd.DataFrame(l)

def check_sign(k):
	if k<0:
		return 'negative'
	else:
		return 'positive'

def get_factors(mapping,gen,complete,binary,predictions,opt='Macro'):
	aux=mapping[opt].values
	l=[]
	for i,value in enumerate(mapping['Values']):
		s = value[:value.index(" ---")]
		v = value[value.index("--- ")+4:]
		o = aux[i]
		l.append([s,v,o])
	mapping=pd.DataFrame(l)
	mapping.columns = ['source','variable','output']



	gen['variable-value'] = gen['variable'] + [' - '] + gen['value']
	match = binary*gen['weight'].values
	match = match.T
	match['variable'] = gen['variable'].values
	match = match.groupby('variable').mean().reset_index()

	gen['variable-value'] = gen['variable'] + [' - '] + gen['value']
	match = binary*gen['weight'].values
	match = match.T
	match['variable'] = gen['variable'].values
	match = match.groupby('variable').mean().reset_index()

	map_dict = (mapping[~mapping.duplicated()][['variable','output']]
				.set_index('variable')
				.to_dict(orient='dict')['output'])

	l=[]
	for k in match['variable']:
		try:
			l.append(map_dict[k])
		except Exception as e:
			#fix this!!
			l.append('Out')
			#print str(e)
	match[opt] = l

	factors = (match.drop(['variable'],axis=1).groupby(opt)
						.mean().T
						.drop(['Out'],axis=1).T)
	factors.columns = complete['global id'].values.astype(str)

	df1=factors.copy(deep=True)
	df2=factors.copy(deep=True)
	for col in factors.columns:
		vector = factors[col]
		df1[col]=[(100*abs(k)/sum(vector.apply(abs))) for k in vector]
		df2[col] = ['negative' if k<0 else 'positive' for k in vector]

	factors = factors.T.reset_index()
	factors_pct = df1.T.reset_index()
	factors_sign = df2.T.reset_index()

	factors_sign = factors_sign.rename(columns={'index':'global id'})
	factors_sign.columns = [k.lower() for k in factors_sign.columns]
	factors_sign['global id'] = factors_sign['global id'].astype(str)

	factors_pct = factors_pct.rename(columns={'index':'global id'})
	predictions['global id'] = predictions['global id'].astype('str')
	factors_pct = factors_pct.join(predictions.set_index('global id'),on='global id')
	factors_pct.columns = [k.lower() for k in factors_pct.columns]
	factors_pct['global id'] = factors_pct['global id'].astype(str)
	
	return factors, factors_pct, factors_sign

def logic_qs(row,main_factor,sub,specifics,questions_df):
	'''
	row - represents the information for one employee
	main_factor - represents the main factor being analized
	sub - the list of subfactors associated with the main factor
	specifics - the list of subfactors that are used in specific rules
	questions_df - the dataframe cointaining the questions mapped to the factors and subfactors
	'''
	
	#get the name of the subfactor with highest weight value
	top_group = row[sub].sort_values(ascending=False).index[0]
	#get the weight of the subfactor with highest weight value
	top_factor = row[sub].sort_values(ascending=False)[0]

	if top_group!='tenure manager':
		#get the sub dataframe with only questions for the top subfactor
		qs=questions_df[questions_df['sub-factor']==top_group].reset_index()
		#select a random question from the sub dataframe
		r2 = random.randint(0,len(qs)-1)
		q2 = qs.ix[r2,'action type']
	else:
		#get the sub dataframe with only questions for the main factor
		qs=questions_df[questions_df['roll-up factor']==main_factor]
		#remove the leading question from the sub dataframe
		qs=qs[qs['sub-factor']!='leading'].reset_index()
		#select a random question from the sub dataframe
		r2 = random.randint(0,len(qs)-1)
		q2 = qs.ix[r2,'action type']

	if top_factor>50:
		#get the sub dataframe with only questions for the top subfactor
		qs=questions_df[questions_df['sub-factor']==top_group].reset_index()
	else:
		idx=1
		#get the sub dataframe with only questions for the next top subfactor
		next_group=row[sub].sort_values(ascending=False).index[idx]
		qs=questions_df[questions_df['sub-factor']==next_group].reset_index()

		while len(qs)==0 and idx<len(row[sub]):
			idx+=1
			#get the sub dataframe with only questions for the next top subfactor
			next_group=row[sub].sort_values(ascending=False).index[idx]
			qs=questions_df[questions_df['sub-factor']==next_group].reset_index()
	
	r_list = np.arange(0,len(qs))
	np.random.shuffle(r_list)
	r_list=list(r_list)

	for r3 in r_list:
		q3 = qs.ix[r3,'action type']
		if q2!=q3:
			break
	
	return q2,q3

def get_confusion_metrics(y_test,pred):
	l=[]
	for i in range(len(y_test)):
		if y_test[i]==1 and y_test[i]==pred[i]:
			l.append('tp')
		elif y_test[i]==0 and y_test[i]==pred[i]:
			l.append('tn')
		elif y_test[i]==1 and y_test[i]!=pred[i]:
			l.append('fn')
		else:
			l.append('fp') 
	return l

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
	return ''.join(random.choice(chars) for _ in range(size))

def get_icons_attr(weights,signs,sign):
	pct=[]
	fact=[]
	for factor in factors:
		factor=factor.replace('_',' ')
		if signs[factor] == sign:
			fact.append(factor)
			pct.append(weights[factor])

	idx=np.argsort(pct)[::-1]
	pct=pd.Series(pct)[idx].values.tolist()
	fact=pd.Series(fact)[idx].values

	attributes=[]
	for i,f in enumerate(fact):
		f=f.replace(' ','_')
		icon = "glyphicon glyphicon-"+factors_icon[f]
		name = f.replace("_"," ").title()
		weight = pct[i]
		attributes.append({'name':name,'icon':icon,'weight':weight})
	return attributes