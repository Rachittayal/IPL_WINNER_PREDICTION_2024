
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("C:\\Users\\rachi\\OneDrive\\Desktop\\Practise Projects\\ipl_2024_deliveries.csv")

df['striker'].head(100)

### This help me to find the coorelations thats why I am removing those columns having dtype=Obj
df_copy=df.copy()
for i in df_copy.columns[0:19]:
    if df_copy[i].dtype=='O':
        df_copy=df_copy.drop(i,axis=1)

import seaborn as sns
sns.heatmap(df.isnull())

###Finding how many null are there
df.isnull().sum()

###Removing null point
df['wicket_type'] = df['wicket_type'].fillna('NO')
df['player_dismissed'] = df['player_dismissed'].fillna('NO')
sns.heatmap(df.isnull())

df.describe()
"""**Now its time perform some Feature Engineering To know the insights**"""

### Finding Total Runs by each Player
total_runs=df.groupby(['striker','match_id','batting_team'])
total_runs_by_each_player=total_runs['runs_of_bat'].sum()
tr=total_runs_by_each_player.to_frame()

tr_sorted = tr.sort_values(by='runs_of_bat', ascending=False)

###Total balls Played by each player
total_balls_played = df.groupby(['striker','match_id','batting_team'])
total_balls_played_by_each=total_balls_played['over'].count()
tb=total_balls_played_by_each.to_frame()

new_df=pd.merge(tr,tb,on=['striker','match_id','batting_team'])

###Now we are finding strike rate of each player
new_df['strike_rate']=(new_df['runs_of_bat'] / new_df['over'])*100

"""### **Inning Wise** *italicised text*"""

###match wise no of four
subset_df = df[df['match_id'].between(202401, 202461)]
fours_df = subset_df[subset_df['runs_of_bat'] == 4]
fours_count = fours_df.groupby(['match_id', 'batting_team','striker']).size().reset_index(name='no_of_fours')

###match wise no of sixes
subset_df = df[df['match_id'].between(202401, 202461)]
sixes_df = subset_df[subset_df['runs_of_bat'] == 6]
sixes_count = sixes_df.groupby(['match_id', 'batting_team','striker']).size().reset_index(name='no_of_sixes')

new_df = pd.merge(new_df, fours_count, on=['batting_team', 'match_id','striker'])
new_df = pd.merge(new_df, sixes_count, on=['batting_team', 'match_id','striker'])

###total no of four
subset_df = df[df['match_id'].between(202401, 202461)]
fours_df = subset_df[subset_df['runs_of_bat'] == 4]
total_fours_count = fours_df.groupby(['batting_team','striker']).size().reset_index(name='no_of_fours')

###total no of sixes
subset_df = df[df['match_id'].between(202401, 202461)]
sixes_df = subset_df[subset_df['runs_of_bat'] == 6]
total_sixes_count = sixes_df.groupby(['batting_team','striker']).size().reset_index(name='no_of_sixes')

total_runs=df.groupby(['striker','innings'])
total_runs_by_each_player_in=total_runs['runs_of_bat'].sum()
trii=total_runs_by_each_player_in.to_frame()

###Total balls Played by each player per innings
total_balls_played = df.groupby(['striker','innings'])
total_balls_played_by_each_in=total_balls_played['over'].count()
tbii=total_balls_played_by_each_in.to_frame()

new_df_inning=pd.merge(trii,tbii,on=['striker','innings'])

###Now we are finding strike rate of each player
new_df_inning['strike_rate_per_innings']=(new_df_inning['runs_of_bat'] / new_df_inning['over'])*100

df['wicket'] = df['wicket_type'].apply(lambda x: 0 if x == 'NO' else 1)

###Total runs of bat given by bowler
total_runsofbat_given = df.groupby(['bowling_team','match_id','bowler'])
total_runsofbat_given_by_each_in=total_runsofbat_given['runs_of_bat'].sum()
trob=total_runsofbat_given_by_each_in.to_frame()

###Total extras  given by bowler
total_extras_given = df.groupby(['bowling_team','match_id','bowler'])
total_extras_given_by_each_in=total_extras_given['extras'].sum()
te=total_extras_given_by_each_in.to_frame()

new_df_bowler = pd.merge(trob, te, on=['bowling_team','match_id', 'bowler'])

new_df_bowler['total_runs_given']=new_df_bowler['runs_of_bat'] + new_df_bowler['extras']

##Total balls count by bowler
bb = df.groupby(['bowling_team','match_id', 'bowler']).size().reset_index(name='total_balls_bowled')

new_df_bowler=pd.merge(trob,te,on=['bowling_team','match_id', 'bowler'])
new_df_bowler_final=pd.merge(new_df_bowler,bb,on=['bowling_team','match_id', 'bowler'])
new_df_bowler_final['total_runs_given']=new_df_bowler_final['runs_of_bat'] + new_df_bowler_final['extras']
new_df_bowler_final['total_overs_bowled']=new_df_bowler_final['total_balls_bowled'] / 6

###Finding Economy of bowler
new_df_bowler_final['Economy rate']=new_df_bowler_final['total_runs_given'] / new_df_bowler_final['total_overs_bowled']

###No of wickets taken
total_wickets=df.groupby(['bowling_team','match_id', 'bowler'])
total_wickets_byeach=total_wickets['wicket'].sum()
tw=total_wickets_byeach.to_frame()

bowlers_stats = pd.merge(new_df_bowler_final, tw, on=['bowling_team','match_id', 'bowler'])
match_wise_stats = pd.merge(new_df, bowlers_stats, on=['match_id'])

"""### **INNING WISE**"""

###Total runs of bat given by bowler per innings
total_runsofbat_given = df.groupby(['bowler','innings'])
total_runsofbat_given_by_each_in=total_runsofbat_given['runs_of_bat'].sum()
trob_in=total_runsofbat_given_by_each_in.to_frame()

###Total extras  given by bowler per innings
total_extras_given = df.groupby(['bowler','innings'])
total_extras_given_by_each_in=total_extras_given['extras'].sum()
te_in=total_extras_given_by_each_in.to_frame()

new_df_bowler_in=pd.merge(trob_in,te_in,on=['bowler','innings'])

new_df_bowler_in['total_runs_given_in']=new_df_bowler_in['runs_of_bat'] + new_df_bowler_in['extras']

# Total balls count by bowler per innings
bb_in = df.groupby(['innings', 'bowler']).size().rename('total_balls_bowled_in')

new_df_bowler_in=pd.merge(trob_in,te_in,on=['bowler','innings'])
new_df_bowler_final_in=pd.merge(new_df_bowler_in,bb_in,on=['bowler','innings'])
new_df_bowler_final_in['total_runs_given_in']=new_df_bowler_final_in['runs_of_bat'] + new_df_bowler_final_in['extras']
new_df_bowler_final_in['total_overs_bowled_in']=new_df_bowler_final_in['total_balls_bowled_in'] / 6
new_df_bowler_final_in

###Finding Economy of bowler
new_df_bowler_final_in['Economy rate_in'] = new_df_bowler_final_in['total_runs_given_in'] / new_df_bowler_final_in['total_overs_bowled_in']
new_df_bowler_final_in

###No of wickets taken
total_wickets_in=df.groupby(['bowler','innings'])
total_wickets_byeach_in=total_wickets_in['wicket'].sum()
tw_in=total_wickets_byeach_in.to_frame()
tw_in

bowlers_stats_in=pd.merge(new_df_bowler_final_in,tw_in,on=(['bowler','innings']))
bowlers_stats_in

df

"""### **TEAM WISE**"""

###total runs by perticular team
team_wise_runs=df.groupby(['match_id','innings','match_no','batting_team','bowling_team'])
team_wise_runs_total=team_wise_runs['runs_of_bat'].sum() + team_wise_runs['extras'].sum()
twtr=team_wise_runs_total.to_frame()
twtr



data = {
    202401: 'CSK',
    202402: 'PBKS',
    202403: 'KKR',
    202404: 'RR',
    202405: 'GT',
    202406: 'RCB',
    202407: 'CSK',
    202408: 'SRH',
    202409: 'RR',
    202410: 'KKR',
    202411: 'LSG',
    202412: 'GT',
    202413: 'DC',
    202414: 'RR',
    202415: 'LSG',
    202416: 'KKR',
    202417: 'PBKS',
    202418: 'SRH',
    202419: 'RR',
    202420: 'MI',
    202421: 'LSG',
    202422: 'CSK',
    202423: 'SRH',
    202424: 'GT',
    202425: 'MI',
    202426: 'DC',
    202427: 'RR',
    202428: 'KKR',
    202429: 'CSK',
    202430: 'SRH',
    202431: 'RR',
    202432: 'DC',
    202433: 'MI',
    202434: 'LSG',
    202435: 'SRH',
    202436: 'KKR',
    202437: 'GT',
    202438: 'RR',
    202439: 'LSG',
    202440: 'DC',
    202441: 'RCB',
    202442: 'PBKS',
    202443: 'DC',
    202444: 'RR',
    202445: 'RCB',
    202446: 'CSK',
    202447: 'KKR',
    202448: 'LSG',
    202449: 'PBKS',
    202450: 'SRH',
    202451: 'KKR',
    202452: 'RCB',
    202453: 'CSK',
    202454: 'KKR',
    202455: 'MI',
    202456: 'DC',
    202457: 'SRH',
    202458: 'RCB',
    202459: 'GT',
    202460: 'KKR',
    202461: 'CSK',
    202462: 'RCB',
    202463: 'NO_WIN',
    202464: 'DC',
    202465: 'PBKS',
    202466: 'NO_WIN',
    202467: 'LSG'
}
twtr = pd.DataFrame(data.items(), columns=['match_id', 'match_winner'])
twtr['match_winner'] = twtr['match_id'].map(data)


twtr = pd.DataFrame(data.items(), columns=['match_id', 'match_winner'])

team_wins = twtr['match_winner'].value_counts().reset_index()
team_wins.columns = ['match_winner', 'wins']

final = pd.merge(twtr, team_wins, on='match_winner')
final.groupby(['match_id']).sum()

final['Total_points']=final['wins']*2
match_wise_stats = pd.merge(match_wise_stats, final, on=['match_id'])
match_wise_stats

data2 = {'match_id':[202401,
    202402,
    202403,
    202404,
    202405,
    202406,
    202407,
    202408,
    202409,
    202410,
    202411,
    202412,
    202413,
    202414,
    202415,
    202416,
    202417,
    202418,
    202419,
    202420,
    202421,
    202422,
    202423,
    202424,
    202425,
    202426,
    202427,
    202428,
    202429,
    202430,
    202431,
    202432,
    202433,
    202434,
    202435,
    202436,
    202437,
    202438,
    202439,
    202440,
    202441,
    202442,
    202443,
    202444,
    202445,
    202446,
    202447,
    202448,
    202449,
    202450,
    202451,
    202452,
    202453,
    202454,
    202455,
    202456,
    202457,
    202458,
    202459,
    202460,
    202461,
    202462,
    202463,
    202464,
    202465,
    202466,
    202467],
         'H&A':['HOME',
    'HOME',
    'HOME',
    'HOME',
    'HOME',
    'HOME',
    'HOME',
    'HOME',
    'HOME',
    'AWAY',
    'HOME',
    'HOME',
    'HOME',
    'AWAY',
    'AWAY',
    'AWAY',
    'AWAY',
    'HOME',
    'HOME',
    'HOME',
    'HOME',
    'HOME',
    'AWAY',
    'AWAY',
    'HOME',
    'AWAY',
    'AWAY',
    'HOME',
    'AWAY',
    'AWAY',
    'AWAY',
    'AWAY',
    'AWAY',
    'HOME',
    'AWAY',
    'HOME',
    'AWAY',
    'HOME',
    'AWAY',
    'HOME',
    'AWAY',
    'AWAY',
    'HOME',
    'AWAY',
    'AWAY',
    'HOME',
    'HOME',
    'HOME',
    'AWAY',
    'HOME',
    'AWAY',
    'HOME',
    'AWAY',
    'AWAY',
    'HOME',
    'HOME',
     'HOME',
    'AWAY',
    'HOME',
     'HOME',
    'HOME',
    'HOME',
    'NO',
    'HOME',
    'AWAY',
    'NO',
    'AWAY']
}

h_w=pd.DataFrame(data2)

twtr_new=pd.merge(twtr,h_w,on='match_id')
toss_win=pd.read_csv("C:\\Users\\rachi\\OneDrive\\Desktop\\Practise Projects\\Untitled spreadsheet - Sheet1.csv")

final_toss_details=pd.merge(twtr_new,toss_win,on='match_id')

import pandas as pd
match_wise_stats = pd.merge(match_wise_stats,final_toss_details, on=['match_id'])
match_wise_stats.drop(['match_winner_y'],axis=1)

import matplotlib.pyplot as plt
import seaborn as sns
fig, axes = plt.subplots(5, 5, figsize=(50, 30))
axes = axes.flatten()
features = df.columns.tolist()
if 'match_winner_x' in features:
    features.remove('match_winner_x')
for i, feature in enumerate(features):
    sns.histplot(data=df, x=feature, ax=axes[i], kde=True)
    axes[i].set_title(f'{feature} vs Match Winner')
plt.tight_layout()
plt.show()

import pandas as pd
if 'match_winner_y_encoded' in match_wise_stats.columns:
    match_wise_stats.drop(['match_winner_y_encoded'], axis=1, inplace=True)
match_wise_stats = pd.get_dummies(match_wise_stats, columns=['batting_team', 'striker', 'bowling_team', 'bowler', 'match_winner_x', 'H&A', 'toss_win_team', 'What_they_choose'])
match_wise_stats.info()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(match_wise_stats['match_winner_y'])

X = match_wise_stats.drop('match_winner_y', axis=1)
y = y_encoded

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

predicted_winner_encoded = rf_model.predict(X_test)
predicted_winner = label_encoder.inverse_transform(predicted_winner_encoded)
print("Predicted winner of IPL 2024 is:", predicted_winner[0])