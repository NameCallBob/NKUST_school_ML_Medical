# -*- coding: utf-8 -*-
"""
Created on Fri May  7 19:14:54 2021

@author: HSUAN
"""


import os,multiprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
os.system("cls");

class Main():
	def __init__(self):
		self.csv_path='../den/csv/csvs2' #csv路徑
		self.outpath_csv="../den/csv/csvs2_train"
		self.outpath_csv_test="../den/csv/csvs2_test"
		self.useCore= 8 # 使用核心
		self.data={
			"den2":np.array([]),
			"gen2":np.array([])
		}
		self.fileName={
			"den2":np.array([]),
			"gen2":np.array([])
		}
	def run(self):
		self.readPath()
		print(self.data['den2'].shape)
		self.Standard()
		self.data['den2']=self.data['den2'].reshape(-1,72,3)
		self.data['gen2']=self.data['gen2'].reshape(-1,72,3)
		print(self.data['den2'].shape)
		print(self.data['gen2'].shape)
		self.export()
	def readPath(self):
		args=[];
		for path,folders,files in os.walk(self.csv_path):
			for file in files:
				if "den2" in path:
					label="den2"
				if "gen2" in path:	
					label="gen2"
				args.append([os.path.join(path,file),file,label])
		with multiprocessing.Pool(processes=self.useCore) as pool:
			result=pool.starmap(self.readData,args)
			for row in result:
				if(row[0]=="den2"):
					if(self.data['den2'].shape[0]==0):
						self.data['den2']=np.array(row[1])
					else:
						self.data['den2']=np.vstack((self.data['den2'],row[1])) #vstack 沿著豎直方向將矩陣堆疊起來。
					self.fileName['den2']=np.append(self.fileName['den2'],row[2])
				else:
					if(self.data['gen2'].shape[0]==0):
						self.data['gen2']=np.array(row[1])
					else:
						self.data['gen2']=np.vstack((self.data['gen2'],row[1]))
					self.fileName['gen2']=np.append(self.fileName['gen2'],row[2])
	def readData(self,filePath,fileName,label):
		try:
			df=pd.read_csv(filePath,index_col=0).fillna(-1).values.tolist()
		except Exception as e:
			print("處理失敗：{}=>{}".format(fileName,e))
		return [label,df,fileName]
	def Standard(self):
			data=np.vstack((self.data['den2'],self.data['gen2']))
			#print(len(data))
			MMS=MinMaxScaler().fit(data)
			data=MMS.transform(data) #在Fit的基礎上，進行標準化，降維，歸一化等操作
			self.data['den2']=data[0:self.data['den2'].shape[0]]
			self.data['gen2']=data[self.data['den2'].shape[0]:]
	def export(self):
		for label in self.data:
			k=0
			csv_outpath="{}/{}".format(self.outpath_csv,label)
			csv_outpath_test="{}/{}".format(self.outpath_csv_test,label)
			if(not os.path.isdir(csv_outpath)):
				os.makedirs(csv_outpath)
			for i,data in enumerate(self.data[label]):
				k+=1
				#df=pd.DataFrame(data,columns=['Hematocrit','Platelets','WBC'])
				df=pd.DataFrame(data,columns=['Hematocrit','Platelets','WBC'])
				df = df.replace(0.0,np.nan) #不填值使用
				df = df.dropna(axis=0)
				if(label=='den2'):
					if(k>30):
						df.to_csv("{}/{}".format(csv_outpath,self.fileName[label][i]))
					else:
						df.to_csv("{}/{}".format(csv_outpath_test,self.fileName[label][i]))
				else:
					if(k>173):
						df.to_csv("{}/{}".format(csv_outpath,self.fileName[label][i]))
					else:
						df.to_csv("{}/{}".format(csv_outpath_test,self.fileName[label][i]))
if __name__=="__main__":
	main=Main()
	main.run()

