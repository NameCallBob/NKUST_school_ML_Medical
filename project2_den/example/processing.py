# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 19:35:34 2021

@author: HSUAN
"""


import pandas as pd
import numpy as np
import os,multiprocessing,datetime,json
from dateutil.parser import parse
useCore= 4 # 使用核心
filter_day=3 # 篩選天數
filter_biomarker_num=2 #篩選一個biomarker測量次數
data_path = "./data" #資料夾目錄
outpath_csv="./csv/all_data"
def Main():
    process(r"C:/Users/ASUS/Desktop/den/data/noDen","den0413_gen000000000all.csv")

def process(filePath,fileName):

	print("處理{}".format(fileName));

  #DEN欄位：idcode,opdno,labno,notidate,rcvdat,rcvtm,labsh1it,labnmabv,labresuval
  #FLU欄位：idcode,opdno,labno,ipdat   ,cltdat,clttm,labsh1it,labnmabv,labresuval
  #GEN欄位：idcode,opdno,labno,ipdat   ,cltdat,clttm,labsh1it,labnmabv,labresuval
  
	if ("noDen" in filePath):
		ipDate="ipdat"
		cltDate="cltdat"
		cltTime="clttm"
	else:
		ipDate="notidate"
		cltDate="rcvdat"
		cltTime="rcvtm"
	try:
		usecols=["idcode","opdno","labno",ipDate,cltDate,cltTime,"labsh1it","labnmabv","labresuval"] # 使用欄位
		df=pd.read_csv(filePath+ "\\" + fileName)
        # 將dgdat改為disdat
		if(ipDate=="notidate"):
			df=df.rename(columns={"notidate": "ipdat"})
		if(cltDate=="rcvdat"):
			df=df.rename(columns={"rcvdat": "cltdat"})
		if(cltTime=="rcvtm"):
			df=df.rename(columns={"rcvtm": "clttm"})
		# 將Biomarker改為名稱
		df=df[df['labsh1it'].isin(['72A001','72B703','72A015','72-547','72C015','72I001','72D001','72-517','72-360','72-517','72-361','72-518','72-333','72-505'])]
		for biomarker in df['labsh1it'].unique():
			if(biomarker=="72A001" or biomarker=="72B703"):
				df.loc[df['labsh1it']==biomarker,'labsh1it']='WBC'
			elif(biomarker=="72A015"):
				df.loc[df['labsh1it']==biomarker,'labsh1it']='Segment'
			elif(biomarker=="72-547"):
				df.loc[df['labsh1it']==biomarker,'labsh1it']='CRP'
			elif(biomarker=="72C015"):
				df.loc[df['labsh1it']==biomarker,'labsh1it']='Lymphocyte'
			elif(biomarker=="72I001"):
				df.loc[df['labsh1it']==biomarker,'labsh1it']='Platelets'
			elif(biomarker=="72-517" or biomarker=="72-360" or biomarker=="72-517"):
				df.loc[df['labsh1it']==biomarker,'labsh1it']='AST/GOT'
			elif(biomarker=="72D001"):
				df.loc[df['labsh1it']==biomarker,'labsh1it']='Hematocrit'
			elif(biomarker=="72-361" or biomarker=="72-518"):
				df.loc[df['labsh1it']==biomarker,'labsh1it']='ALT/GPT'
			elif(biomarker=="72-333" or biomarker=="72-505"):
				df.loc[df['labsh1it']==biomarker,'labsh1it']='Creatinine'
		# 移除 Lactate
		#df=df[df['labsh1it']!='Segment']

        #日期格式轉換
		df = df.dropna()
		df["clttm"] = pd.to_numeric(df.clttm, errors='coerce')

		df['clttm'] = df['clttm'].fillna(0).astype('int32')
		df['cltdat'] = df['cltdat'].fillna(0).astype('int32')

		df["ipdat"]=df["ipdat"].apply(lambda x: datetime.date(int(str(x)[0:4]),int(str(x)[4:6]),int(str(x)[6:8])))
		df["cltdat"]=df["cltdat"].apply(lambda x: datetime.date(int(str(x)[0:4]),int(str(x)[4:6]),int(str(x)[6:8])))
		df['clttm']=df['clttm'].apply(lambda x: str(x) if len(str(x))==4 else '0'+str(x))
		df['clttm']=df['clttm'].apply(lambda x: str(x) if len(str(x))==4 else '0'+str(x))
		df['clthr']=df['clttm'].apply(lambda x: x[0:2])
		df['cltmin']=df['clttm'].apply(lambda x: x[2:4])
		df['labresuval']=pd.to_numeric(df['labresuval'], errors='coerce') # 將Biomarker的值改為float，並且將非float改為NaN(coerce)
		df['clt_day']=(pd.to_datetime(df["cltdat"])-pd.to_datetime(df['ipdat'])).apply(lambda x: x.days)
		df=(df[df['ipdat']<=df['cltdat']]) #檢測時間超過輸入時間
		df=(df[df['clt_day']<3]) #檢測時間要在三天內 do
		df.to_csv("./df.csv")
		#建立PivotTable index(欄位) 將labsh1it檢查項目列為columes，value=檢查次數
		df_pivot=df.pivot_table(index=["idcode","opdno","ipdat"],columns='labsh1it',values='labresuval',aggfunc=lambda x: len(x.unique())).fillna(0).astype(int).rename_axis(None, axis=1).reset_index()
		df_pivot.to_csv("./createPivot_1.csv")
		#print(df_pivot['Segment'].value_counts())
		# 日期轉換、入院到出院天數計算
		#df_pivot['day']=(pd.to_datetime(df_pivot["disdat"])-pd.to_datetime(df_pivot['ipdat'])).apply(lambda x: x.days)
		#df_pivot.to_csv("./createPivot_day.csv")
		# 篩選不滿filter_day(3天)
		#df_pivot=(df_pivot[df_pivot['day']>=filter_day])
		#df_pivot=df_pivot.drop(['day'], axis=1) #drop不需要用到的欄位
		biomarkers=['Hematocrit','Platelets','WBC'] # 所有biomarker
        #將biomarker測量次數沒有大於兩次的去除
		for biomarker in biomarkers:
			print(biomarker)
			print(df_pivot[df_pivot[biomarker]>=2])
			df_pivot=df_pivot[df_pivot[biomarker]>=2]
		label="gen2"
		df_pivot.to_csv("./createPivot_2.csv")
		csv_outpath="{}/{}".format(outpath_csv,label)
		if(not os.path.isdir(csv_outpath)):
			os.makedirs(csv_outpath)
		for i,row in df_pivot.iterrows():
			opdno_df=(df[(df['idcode']==row['idcode']) & (df['opdno']==row['opdno']) & (df['ipdat']==row['ipdat']) ])
			opdno_df=opdno_df.pivot_table(index=["idcode","opdno","ipdat","cltdat","clthr"],columns='labsh1it',values='labresuval').rename_axis(None, axis=1).reset_index()
			new_opdno_df=pd.DataFrame(None,columns=['Hematocrit','Platelets','WBC'])
			for date in [row['ipdat'],row['ipdat']+datetime.timedelta(days=1),row['ipdat']+datetime.timedelta(days=2)]:
				for hour in range(0,24):
					tmp=opdno_df[(opdno_df['cltdat']==date) & (opdno_df['clthr']==(str(hour) if len(str(hour))==2 else '0'+str(hour)))].loc[:,['Hematocrit','Platelets','WBC']];
					if(tmp.shape[0]==0):
						new_opdno_df=new_opdno_df.append(pd.Series([]), ignore_index=True)
					else:
						new_opdno_df=new_opdno_df.append(tmp, ignore_index=True)
			#new_opdno_df.to_csv("{}/{}_{}.csv".format(csv_outpath,row["idcode"],row["opdno"]))
			new_opdno_df=new_opdno_df.fillna(method='ffill').fillna(method='bfill').astype("float16")
			new_opdno_df=new_opdno_df.replace([np.inf, -np.inf],0)
			new_opdno_df.to_csv("{}/{}_{}.csv".format(csv_outpath,row["idcode"],row["opdno"]))

		print('{}處理完成'.format(fileName))
	except Exception as e:
		print("處理失敗：{}=>{}".format(fileName,e))

if __name__=="__main__":
    Main()
    print('end')

