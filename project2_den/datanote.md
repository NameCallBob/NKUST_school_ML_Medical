資料說明：  

========

*   den00000 ＝ 登革熱患者  
    
*   flu00000 ＝ 流感患者
    
*   gen0000、gen0001 ＝ 菌血症患者
    
*   sep0000 ＝ 敗血症患者
    

  

欄位說明：72小時資料，三天紀錄
================

  
loc -> 表院區

idcode -> 病歷號碼  

opdno -> 門診號碼

csn -> 住院號

admdat -> 入院日期

bdate -> 病患生日

sex ->  性別

notidate -> 檢查日期

sourno -> 檢查大項

labno -> 檢查單編號  

rcvdat -> 檢查日期

rcvtm -> 檢查的時間

labit -> 檢查項目編號

labsh1it -> 檢查編號

labnmabc -> 檢查項目名稱

labtrfvbsl -> 檢查項目標準值

labresuval -> 檢查結果

  
處理方式  
使用者為idcode opdno為就診編號依此進行分組，查看該次就診編號他的項目及結果，最後補資料進行，補資料意旨每小時都會有測量資料，假如3跟8點有資料，4567沒有，他們必須要參照3的資料．  
  
目標  
W1 預測登革熱及非登革熱

W2 登革熱、流感、敗血症、菌血症

  
記得列出訓練和驗證績效