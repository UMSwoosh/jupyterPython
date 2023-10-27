#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import nfl_data_py as nfl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.display import display, HTML
######
import datetime as dt
import time
display(HTML("<style>.container { width:100% !important; }</style>"))
InteractiveShell.ast_node_interactivity = "all"
pd.set_option('display.max_colwidth', 400)
pd.set_option('display.max_columns', None)


# # Mapping/Variable Creation

# In[49]:


#mapping = pd.read_excel('M:/pythFiles/Football/teamMappings.xlsx')
t = dt.datetime.today().strftime("%Y%m%d")
baseDir = 'M:/pythFiles/Football/2023/'

#Week number of the year (Monday as the first day of the week) as a decimal number [00,53]. 
#All days in a new year preceding the first Monday are considered to be in week 0.
year = '2023'
yearWeek = int(time.strftime('%Y%W'))
yearFactor = int(time.strftime('%Y'))*100
###Current Week Variables
weekVarDec = yearWeek - yearFactor - 35
weekVarAsAlpha1 = str(weekVarDec)
weekVarWeekW1dig = 'week' + weekVarAsAlpha1
###Handle 0 Pad for Current Week
if weekVarDec < 10:
    weekVarAsAlpha2 = '0' + weekVarAsAlpha1
    weekVarWeekW2dig = 'week' + weekVarAsAlpha2
else:
    weekVarAsAlpha2 = weekVarDec
    weekVarWeekW2dig = 'week' + weekVarAsAlpha1
###Previous Week Variables
weekVarDecPvs = yearWeek - yearFactor - 36
weekVarAsAlpha1Pvs = str(weekVarDecPvs)
weekVarWeekW1digPvs = 'week' + weekVarAsAlpha1Pvs
###Handle 0 Pad for Previous Week
if weekVarDecPvs < 10:
    weekVarAsAlpha2Pvs = '0' + weekVarAsAlpha1Pvs
    weekVarWeekW2digPvs = 'week' + weekVarAsAlpha2Pvs
else:
    weekVarAsAlpha2Pvs = weekVarDecpvs
    weekVarWeekW2digPvs = 'week' + weekVarAsAlpha1Pvs
###2 Week Prior
weekVarDecPvs2 = yearWeek - yearFactor - 37
weekVarAsAlpha1Pvs2 = str(weekVarDecPvs2)
weekVarWeekW1digPvs2 = 'week' + weekVarAsAlpha1Pvs2
###Handle 0 Pad for Previous Week
if weekVarDecPvs2 < 10:
    weekVarAsAlpha2Pvs2 = '0' + weekVarAsAlpha1Pvs2
    weekVarWeekW2digPvs2 = 'week' + weekVarAsAlpha2Pvs2
else:
    weekVarAsAlpha2Pvs2 = weekVarDecpvs2
    weekVarWeekW2digPvs2 = 'week' + weekVarAsAlpha1Pvs2
print('The current week is:  ' + weekVarAsAlpha2 )
print('The previous week is:  ' + weekVarAsAlpha2Pvs )
print('Two weeks ago is:  ' + weekVarAsAlpha2Pvs2 )


# # Chapter 1

# In[2]:


pbp_py = nfl.import_pbp_data([2021])


# In[4]:


filter_crit = 'play_type == "pass" & air_yards.notnull()'
pbp_py_p = (pbp_py.query(filter_crit).groupby(["passer_id","passer"]).agg({"air_yards":["count","mean"]}))


# In[7]:


pbp_py_p.columns = list(map("_".join, pbp_py_p.columns.values))
sort_crit = "air_yards_count > 100"
print(pbp_py_p.query(sort_crit).sort_values(by="air_yards_mean", ascending=[False]).to_string())


# # Chapter 2

# In[9]:


seasons = range(2016,2022 + 1)
pbp_py = nfl.import_pbp_data(seasons)


# In[10]:


pbp_py_p = pbp_py.query("play_type == 'pass' & air_yards.notnull()").reset_index()


# In[11]:


pbp_py_p["pass_length_air_yards"] = np.where(pbp_py_p["air_yards"] >= 20, "long","short")


# In[12]:


pbp_py_p["passing_yards"] = np.where(pbp_py_p["passing_yards"].isnull(),0,pbp_py_p["passing_yards"])


# In[13]:


pbp_py_p["passing_yards"].describe()


# In[14]:


pbp_py_p.query('pass_length_air_yards == "short"')["passing_yards"].describe()


# In[15]:


pbp_py_p.query('pass_length_air_yards == "long"')["passing_yards"].describe()


# In[16]:


pbp_py_p.query('pass_length_air_yards == "short"')["epa"].describe()


# In[17]:


pbp_py_p.query('pass_length_air_yards == "long"')["epa"].describe()


# In[19]:


sns.displot(data=pbp_py, x="passing_yards");
plt.show();


# In[21]:


sns.set_theme(style="whitegrid", palette="colorblind")
pbp_py_p_short = pbp_py_p.query('pass_length_air_yards == "short"')


# In[22]:


pbp_py_hist_short = sns.displot(data=pbp_py_p_short, binwidth=1, x="passing_yards");pbp_py_hist_short.set_axis_labels("Yards gained (or lost) during a passing play","Count");plt.show();


# In[ ]:


# begin page 36


# # Defense Analysis 

# ## Tackles & Assists Data

# In[2]:


df = nfl.import_pbp_data([2023])
df1 = df.loc[df['play_type'].isin(['pass', 'run'])].reset_index()


# In[69]:


s1 = df1[['week', 'game_id', 'solo_tackle_1_team', 'solo_tackle_1_player_name', 'solo_tackle_1_player_id']].copy()
s1 = s1.dropna (axis=0, subset=['solo_tackle_1_player_name'])
s1['tackles'] = 1
s1.columns = ['week', 'game_id', 'solo_tackle_team', 'solo_tackle_player_name', 'solo_tackle_player_id', 'tackles']
s2 = df1[['week','game_id', 'solo_tackle_2_team','solo_tackle_2_player_name', 'solo_tackle_2_player_id']].copy()
s2 =  s2.dropna (axis=0, subset=['solo_tackle_2_player_name'])
s2['tackles'] = 1
s2.columns = ['week', 'game_id', 'solo_tackle_team', 'solo_tackle_player_name', 'solo_tackle_player_id', 'tackles']
tmp = pd.concat([s1,s2])
tmp.head()


# In[70]:


#Get counts each game
solo = tmp.groupby(['solo_tackle_player_name', 'solo_tackle_team', 'solo_tackle_player_id', 'game_id']).agg({'tackles': ['count']})
#Drop multi-index
solo = solo.reset_index()
#rename columns
solo.columns = ['solo_tackle_player_name','solo_tackle_team', 'solo_tackle_player_id', 'game_id', 'solo_tackles']
#Get Average by player
soloA = solo.groupby(['solo_tackle_player_name', 'solo_tackle_player_id']).agg({'solo_tackles': ['mean']})
soloA = soloA.reset_index()
soloA.columns = ['solo_tackle_player_name', 'solo_tackle_player_id', 'solo_tackles_mean']
#Drop player Name
soloT = soloA[['solo_tackle_player_id', 'solo_tackles_mean']].copy() 
#Merge the Mean
soloFinal = pd.merge(solo, soloT, left_on='solo_tackle_player_id', right_on='solo_tackle_player_id', how='left')
soloFinal


# In[71]:


ta1 = df1[['week', 'game_id', 'tackle_with_assist_1_team', 'tackle_with_assist_1_player_name', 'tackle_with_assist_1_player_id']].copy() 
ta1 = ta1.dropna (axis=0, subset=['tackle_with_assist_1_player_name'])
ta1['tacklewithAssists'] = 1
ta1.columns = ['week', 'game_id', 'tackle_with_assist_team', 'tackle_with_assist_player_name', 'tackle_with_assist_player_id', 'tacklewithAssists']
ta2 = df1[['week','game_id', 'tackle_with_assist_2_team', 'tackle_with_assist_2_player_name', 'tackle_with_assist_2_player_id']].copy()
ta2 = ta2.dropna (axis=0, subset=['tackle_with_assist_2_player_name'])
ta2['tackleWithAssists'] = 1
ta2.columns = ['week', 'game_id', 'tackle_with_assist_team', 'tackle_with_assist_player_name', 'tackle_with_assist_player_id', 'tacklewithAssists']
tmp3 = pd.concat([ta1, ta2])
tmp3.head (10)


# In[72]:


#Get counts each game
twaFinal = tmp3.groupby(['tackle_with_assist_player_name', 'tackle_with_assist_player_id', 'game_id']).agg({'tacklewithAssists' : ['count']})
#Drop multi-index
twaFinal = twaFinal.reset_index()
#rename columns
twaFinal.columns = ['tackle_with_assist_player_name', 'tackle_with_assist_player_id', 'game_id', 'tacklewithAssists']
twaFinal


# In[74]:


comb


# In[75]:


#Combine solo and tackle with
comb =  pd.merge(soloFinal, twaFinal, left_on=['solo_tackle_player_id', 'game_id'], right_on=['tackle_with_assist_player_id', 'game_id'],how='left')
comb['tacklewithAssists'] =  comb['tacklewithAssists'].fillna(0)
comb['combined'] = comb['solo_tackles'] + comb['tacklewithAssists']
comb1 = comb[['solo_tackle_player_name', 'solo_tackle_team', 'solo_tackle_player_id', 'game_id', 'solo_tackles', 'solo_tackles_mean', 'tacklewithAssists', 'combined']].copy() 
comb2 =  comb1.groupby(['solo_tackle_player_name', 'solo_tackle_player_id']).agg({ 'combined' : ['mean']})
comb2 = comb2.reset_index()
comb2.columns = ['solo_tackle_player_name', 'solo_tackle_player_id', 'combinedMean']
#Drop player Name
comb3 =  comb2[['solo_tackle_player_id', 'combinedMean']].copy()
#Merge the Mean
comb4 = pd.merge(comb1, comb3, left_on='solo_tackle_player_id', right_on='solo_tackle_player_id', how='left')
comb4


# In[76]:


a1 = df1[['week', 'game_id', 'assist_tackle_1_team', 'assist_tackle_1_player_name', 'assist_tackle_1_player_id']].copy() 
a1 = a1.dropna (axis=0, subset=['assist_tackle_1_player_name'])
a1['assist_tackle'] = 1
a1.columns = ['week', 'game_id', 'assist_tackle_team', 'assist_tackle_player_name', 'assist_tackle_player_id', 'assist_tackle'] 
a2 = df1[['week', 'game_id', 'assist_tackle_2_team', 'assist_tackle_2_player_name', 'assist_tackle_2_player_id']].copy()
a2 = a2.dropna (axis=0, subset=['assist_tackle_2_player_name']) 
a2['assist_tackle'] = 1
a2.columns = ['week', 'game_id', 'assist_tackle_team', 'assist_tackle_player_name', 'assist_tackle_player_id', 'assist_tackle']
a3 = df1[['week', 'game_id', 'assist_tackle_3_team', 'assist_tackle_3_player_name', 'assist_tackle_3_player_id']].copy() 
a3 = a3.dropna (axis=0, subset=['assist_tackle_3_player_name'])
a3['assist_tackle'] = 1
a3.columns =  ['week', 'game_id', 'assist_tackle_team', 'assist_tackle_player_name', 'assist_tackle_player_id', 'assist_tackle']
a4 = df1[['week', 'game_id', 'assist_tackle_4_team', 'assist_tackle_4_player_name', 'assist_tackle_4_player_id']].copy() 
a4 =  a4.dropna(axis=0, subset=['assist_tackle_4_player_name'])
a4['assist_tackle'] = 1
a4.columns  = ['week', 'game_id', 'assist_tackle_team', 'assist_tackle_player_name', 'assist_tackle_player_id', 'assist_tackle']
tmp2 = pd.concat([a1,a2,a3,a4])
tmp2.head()


# In[77]:


ast = tmp2.groupby(['assist_tackle_player_name', 'assist_tackle_player_id', 'game_id']).agg({'assist_tackle': ['count']})
#Drop multi-index 
assist =  ast.reset_index()
#rename columns
assist.columns =  ['assist_tackle_player_name', 'assist_tackle_player_id', 'game_id', 'assist_tackles']
#Get Average by player 
assistA = assist.groupby(['assist_tackle_player_name', 'assist_tackle_player_id']).agg({'assist_tackles': ['mean']}) 
assistA =  assistA.reset_index()
assistA.columns = ['assist_tackle_player_name', 'assist_tackle_player_id', 'assist_tackles_mean']
#Drop player Name 
assistT = assistA[[ 'assist_tackle_player_id', 'assist_tackles_mean']].copy()
#Merge the Mean
assistFinal = pd.merge(assist, assistT, left_on='assist_tackle_player_id', right_on='assist_tackle_player_id')
assistFinal


# In[78]:


#Combine solo and assist
tmpfnl = pd.merge(comb4, assistFinal, left_on=['solo_tackle_player_id', 'game_id'], right_on=[ 'assist_tackle_player_id', 'game_id'], how='left')
fnl = tmpfnl[['solo_tackle_player_name', 'solo_tackle_team', 'solo_tackle_player_id', 'game_id', 'solo_tackles', 'solo_tackles_mean', 'tacklewithAssists', 'combined', 'combinedMean', 'assist_tackles', 'assist_tackles_mean']].copy()
fnl['assist_tackles'] =  fnl['assist_tackles'].fillna(0)
fnl['assist_tackles_mean'] =  fnl['assist_tackles_mean'].fillna(0)
fnl['tackleAssistsCombined'] = fnl['combined'] + fnl['assist_tackles']
fnl['week'] = weekVarDecPvs
gm = pd.DataFrame(fnl.solo_tackle_player_id.value_counts())
gm = gm.reset_index()
gm.columns = ['solo_tackle_player_id', 'gamesPlayed']
final = pd.merge(fnl, gm, left_on='solo_tackle_player_id', right_on='solo_tackle_player_id',how='left')
final


# In[84]:


final.loc[final['solo_tackle_player_name'].isin(['Z.Smith'])]


# ## Import List (In Progress)

# In[85]:


pBase = baseDir + 'playerBase.xlsx'
pMap = baseDir + 'playerMap.xlsx'


# In[86]:


base = pd.read_excel(pBase)
mapping = pd.read_excel(pMap)
base = base.dropna(axis=0, how='any')
base.columns = ['dkPlayer', 'dkOVER', 'dkUNDER']
base = base[base.dkPlayer != 'PLAYER']
base['dkOVER'] = base['dkOVER'].replace({'O': ''}, regex=True)
base['dkOVER'] = base['dkOVER'].str.strip()
base['dkOVER'] = base['dkOVER'].astype(float)
base['dkUNDER'] = base['dkUNDER'].replace({'U': ''}, regex=True)
base['dkUNDER'] = base['dkUNDER'].str.strip()
base['dkUNDER'] = base['dkUNDER'].astype(float)
#### need to solve for when a player is not in a certain prop
base["propType"] = base.groupby("dkPlayer").cumcount()+1
#base = base.reset_index()
len(base)


# In[87]:


dk = pd.merge(mapping, base, left_on='Dkname', right_on='dkPlayer')
dk.head()
#### need to isolate tackle, assist, combo props for compare
dkT = dk.loc[dk['propType'] == 1]
dkA = dk.loc[dk['propType'] == 2]
dkC = dk.loc[dk['propType'] == 3]
len(dkT)
len(dkA)
len(dkC)


# In[271]:


dk.NFLpyName.value_counts()


# In[90]:


dkT1 = pd.merge(final, dkT, left_on='solo_tackle_player_id', right_on='NFLpyID')
dkT1['overFlag'] = np.where(dkT1['dkOVER'] > dkT1['solo_tackles'], 0, 1)
dkT2 = dkT1.groupby(['solo_tackle_player_id']).agg({'overFlag' : ['mean']})
dkTF = pd.merge(dkT1, dkT2, left_on='solo_tackle_player_id', right_on='solo_tackle_player_id')
dkTF.columns = ['solo_tackle_player_name', 'solo_tackle_team', 'solo_tackle_player_id','game_id','solo_tackles','solo_tackles_mean','tacklewithAssists','combined','combinedMean','assist_tackles','assist_tackles_mean','tackleAssistsCombined',
                'week','gamesPlayed','NFLpyName','NFLpyID','Dkname','leagueName','pyTeam','dkPlayer','dkOVER',   'dkUNDER','propType','overFlag','pctGamesOver']


# In[92]:


dkTF.loc[dkTF['pctGamesOver'] == 1]


# In[262]:


dkTF.to_excel(baseDir + "Tackles" + t + ".xlsx")


# In[277]:


dkA1 = pd.merge(assistFinal, dkA, left_on='assist_tackle_player_id', right_on='NFLpyID')
dkA1['overFlag'] = np.where(dkA1['dkOVER'] > dkA1['assist_tackles'], 0, 1)
dkA2 = dkA1.groupby(['assist_tackle_player_id']).agg({'overFlag' : ['mean']})
dkAF = pd.merge(dkA1, dkA2, left_on='assist_tackle_player_id', right_on='assist_tackle_player_id')
dkAF.columns = ['assist_tackle_player_name','assist_tackle_player_id','game_id','assist_tackles','assist_tackles_mean','NFLpyName','NFLpyID','Dkname','leagueName','pyTeam','dkPlayer','dkOVER','dkUNDER','propType','overFlag','pctGamesOver']


# In[278]:


dkAFtmp = dkAF.loc[dkAF['pctGamesOver'] == 1]
dkAFplays = dkAFtmp.groupby(['assist_tackle_player_name','assist_tackle_player_id','assist_tackles_mean','dkOVER','pyTeam']).agg({'game_id' : ['count']})
dkAFplays = dkAFplays.reset_index()
dkAFplays.columns = ['assist_tackle_player_name', 'assist_tackle_player_id', 'assist_tackles_mean', 'dkOVER', 'pyTeam','totGames']
dkAFplays.sort_values('totGames',ascending=False)


# In[279]:


dkAFplays


# In[219]:


dkAF.to_excel(baseDir + "Assists" + t + ".xlsx")


# In[230]:


tmp.to_excel(baseDir + "mapUpdate" + t + ".xlsx")


# ## Defense Boneyard

# In[52]:


len(comb)


# In[49]:


len(assistFinal)


# In[50]:


len(soloFinal)


# In[ ]:





# In[33]:


baseDir = 'M:/pythFiles/Football/2023/'
dee = baseDir + "defense2.xlsx"


# In[5]:


pbp_py.to_excel(dee)


# In[13]:


lst = pd.DataFrame(pbp_py.columns)


# In[14]:


lst.to_excel(dee)


# In[21]:


tack_cols = [col for col in pbp_py.columns if 'tackle' in col]
#print(list(pbp_py.columns))
print(tack_cols)


# In[35]:


tackle = pbp_py[['week', 'game_id', 'solo_tackle', 'tackled_for_loss', 'assist_tackle', 'tackle_for_loss_1_player_id', 'tackle_for_loss_1_player_name', 'tackle_for_loss_2_player_id', 'tackle_for_loss_2_player_name', 'solo_tackle_1_team', 'solo_tackle_2_team', 'solo_tackle_1_player_id', 'solo_tackle_2_player_id', 
                 'solo_tackle_1_player_name', 'solo_tackle_2_player_name', 'assist_tackle_1_player_id', 'assist_tackle_1_player_name', 'assist_tackle_1_team', 'assist_tackle_2_player_id', 'assist_tackle_2_player_name', 'assist_tackle_2_team', 'assist_tackle_3_player_id', 
                 'assist_tackle_3_player_name', 'assist_tackle_3_team', 'assist_tackle_4_player_id', 'assist_tackle_4_player_name', 'assist_tackle_4_team', 'tackle_with_assist', 'tackle_with_assist_1_player_id', 'tackle_with_assist_1_player_name', 'tackle_with_assist_1_team', 
                 'tackle_with_assist_2_player_id', 'tackle_with_assist_2_player_name', 'tackle_with_assist_2_team']].copy()


# In[36]:


len(tackle)
tack = tackle.dropna(axis = 0, how = 'all').copy()
tack['sumCheck'] = tack['solo_tackle'] + tack['tackled_for_loss'] + tack['assist_tackle'] + tack['tackle_with_assist']
tack = tack.loc[tack['sumCheck'] >= 1]
len(tack)


# In[31]:


tack.to_excel(dee)


# In[37]:


tack[tack.isin(['S.Murphy-Bunting']).any(axis=1)].to_excel(dee)


# In[ ]:




