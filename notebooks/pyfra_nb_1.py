# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -LanguageId
#     formats: ipynb,py:light
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
# ---

#
# Notebook 1
# ==============
# Exploring Data and Visualization

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
# %matplotlib inline

# # Import Data

french_categories = {'characteristics': 'caracteristiques', 'places':'lieux', 'users':'usagers', 'vehicles':'vehicules'}
data_categories = french_categories.keys()
categories_dict = dict(zip(data_categories, [0,0,0,0]))


# +
# Define the function that reads the raw data for the specified time range
def read_csv_of_year(start_year, end_year, separators, name_separator='_'):
    if len (separators)<4:
        separators = [separators]*4
        
    df_dict = {}
    for year in range(start_year,end_year+1):
        this_year_str = str(year)
        # Data Category
        this_df_dict = {}        
        for this_category, this_sep in zip(data_categories, separators):
            # We need the French name of the category for the filename
            this_french_category = french_categories[this_category]
            this_file_path_and_name = '../data/'+this_year_str+'/' + this_french_category+name_separator+this_year_str+'.csv'
            this_df_dict[this_category] = pd.read_csv(this_file_path_and_name, encoding='latin-1', sep=this_sep, low_memory=False)
        df_dict[year] = this_df_dict
    return df_dict

# Import years
df_dict = {}
df_dict.update(read_csv_of_year(2005, 2008, separators=','))
df_dict.update(read_csv_of_year(2009,2009, separators=['\t', ',', ',', ',']))
df_dict.update(read_csv_of_year(2010, 2016, separators=','))
df_dict.update(read_csv_of_year(2017, 2018, separators=',', name_separator='-'))
df_dict.update(read_csv_of_year(2019, 2021, separators=';', name_separator='-'))

# -

# ## Put all the data in one dataframe for each category

# +
dict_of_category_dfs = {}
for this_category in data_categories:
    dict_of_category_dfs[this_category] = pd.concat([df_dict[year][this_category] for year in range(2005,2022)], ignore_index=True)

characteristics = dict_of_category_dfs['characteristics']
places = dict_of_category_dfs['places']
users = dict_of_category_dfs['users']
vehicles = dict_of_category_dfs['vehicles']
# -

# # Data Cleaning
# We will perform some of the cleaning of the data on the individual datasets. Not all cleaning is possible before merging the datasets, so there will be a second round of cleaning.

places.columns


# ## Calculate the percentage of missing values for each dataframe

def na_percentage(df):
  return df.isna().sum() *100 / len(df)


for this_category, df in dict_of_category_dfs.items():
    print(this_category+'\n', na_percentage(df),'\n')

# ## Users Dataset

# Dropping unwanted columns , which are num_veh , and id_vehicule
#

users = users.drop(columns=['num_veh','id_vehicule']) #Not needed

# ## Places Dataset

# +
# Change french names against english names (Teamdecision)
# Droped 'Unnamed: 0','v1','v2','vma', because they contained no information.

places = places.drop(['v1','v2','vma','voie','env1'], axis = 1)
places = places.rename(columns = {'catr' : 'Rd_Cat', 'circ' : 'Traf_Direct' , 'nbv' : 'Lanes' ,
                           'pr' : 'Landmark' , 'pr1' : 'Dist_to_Landmark', 'vosp' : 'Add_Lanes', 'prof' : 'Rd_Prof' ,
                          'plan' : 'Rd_Plan' , 'lartpc' : 'Gre_Verge' , 'larrout' : 'Rd_Width', 'surf' : 'Rd_Cond',
                          'infra' : 'Envinmt' , 'situ' : 'Pos_Acc'})
places.head()
# -

# ### Change Nans against zeros

# Set most empty varibles to Zero / Null, because its for all vaiables not in use and can be defined as not applicable.
#
# 9 Variables have <= 1% missing information, so for those it should be fine to set the missing information just tu zero.
# In addition, the recorded data are not suitable for filling the NaNs with, for example, the mean value, since this is almost exclusively about describing states.
#
# Hoped to fill up the missing information for Rd_Width with a comparsion Rd_Nr vs. Rd_Width, but it turns out that the same street has different widths.
#
# Nr_n_Width = places[['Rd_Nr','Rd_Width','Gre_Verge']]#comparsion Rd_Nr vs. Rd_Width. Same for Gre_Verge.
# Nr_n_Width.head()
#
# Landmark and Dist_to_Landmark are information to localize an accident. Nearly 50% of the Data are missing but I will keep the Data. Maybe it will be usefull to complete some location data.
#
# Missing information of Rd_Nr, biggest problem is that later in the Datasets they changed input of numbers against names. So I need a list which says which street is which number. I will drop the variable, it turns out useless.
#
# For column school there are 3 Types of information 99.0 / 0.0 and 3.0, according to the description the variable schools should only contain 1 or 2 so if it is or not near by a school.
# I can't find a logical and reliable way to replace the data. So I will drop them
# In 2019 they droped this column and start with speed limits. Its a importent information but I cant use it in this format. I will drop it for the moment.
# Code
# sns.countplot( x = places.School)

# +

places['Rd_Cat'] = places['Rd_Cat'].fillna(0.0)
places['Traf_Direct'] = places['Traf_Direct'].fillna(0.0)
places['Lanes'] = places['Lanes'].fillna(0.0)
places['Landmark'] = places['Landmark'].fillna(0.0)
places['Dist_to_Landmark'] = places['Dist_to_Landmark'].fillna(0.0)
places['Add_Lanes'] = places['Add_Lanes'].fillna(0.0)
places['Rd_Prof'] = places['Rd_Prof'].fillna(0.0)
places['Rd_Plan'] = places['Rd_Plan'].fillna(0.0)
places['Gre_Verge'] = places['Gre_Verge'].fillna(0.0)
places['Rd_Width'] = places['Rd_Width'].fillna(0.0)
places['Rd_Cond'] = places['Rd_Cond'].fillna(0.0)
places['Envinmt'] = places['Envinmt'].fillna(0.0)
places['Pos_Acc'] = places['Pos_Acc'].fillna(0.0)


# +
# Convert object to float
places['Landmark'] = pd.to_numeric(places['Landmark'],errors = 'coerce')
places['Dist_to_Landmark'] = pd.to_numeric(places['Dist_to_Landmark'],errors = 'coerce')
places['Gre_Verge'] = pd.to_numeric(places['Gre_Verge'],errors = 'coerce')
places['Rd_Width'] = pd.to_numeric(places['Rd_Width'],errors = 'coerce')

# replace empty cells with nans
places.replace('', np.nan)
places = places.copy()

# fill nans with 0
places['Landmark'] = places['Landmark'].fillna(0.0)
places['Dist_to_Landmark'] = places['Dist_to_Landmark'].fillna(0.0)
places['Gre_Verge'] = places['Gre_Verge'].fillna(0.0)
places['Rd_Width'] = places['Rd_Width'].fillna(0.0)

# Convert float to int
places['Rd_Cat'] = places['Rd_Cat'].astype(int, errors = 'raise')
places['Traf_Direct'] = places['Traf_Direct'].astype(int, errors = 'raise')
places['Lanes'] = places['Lanes'].astype(int, errors = 'raise')
places['Landmark'] = places['Landmark'].astype(int, errors = 'raise')
places['Dist_to_Landmark'] = places['Dist_to_Landmark'].astype(int, errors = 'raise')
places['Add_Lanes'] = places['Add_Lanes'].astype(int, errors = 'raise')
places['Rd_Prof'] = places['Rd_Prof'].astype(int, errors = 'raise')
places['Rd_Plan'] = places['Rd_Plan'].astype(int, errors = 'raise')
places['Gre_Verge'] = places['Gre_Verge'].astype(int, errors = 'raise')
places['Rd_Width'] = places['Rd_Width'].astype(int, errors = 'raise')
places['Rd_Cond'] = places['Rd_Cond'].astype(int, errors = 'raise')
places['Envinmt'] = places['Envinmt'].astype(int, errors = 'raise')
places['Pos_Acc'] = places['Pos_Acc'].astype(int, errors = 'raise')

print(places.isna().sum())
print()
print(places.info())
print()
print(places.shape)#it appears that there is a problem with the shape of the df (couldnt normalize) ValueError: Found array with dim 3. the normalize function expected <= 2.

# -

# ## Characteristics Dataset

# ### Translate variable names from French to English

# +
# Translation of the variable nacmes from French to English, also improving the names so that it becomes clearer, what they are about
characteristics.rename(columns={'an': 'year', 'mois':'month', 'jour': 'day', 'hrmn':'hhmm', 
                                'lum': 'daylight', 'agg': 'built-up_area', 'int':'intersection_category', 'atm': 'atmospheric_conditions',
                                'col': 'collision_category', 'com': 'municipality', 'adr':'adress', 'gps': 'gps_origin', 'lat': 'latitude',
                                'long': 'longitude', 'dep': 'department'}, inplace=True)

# Change the values for 'built-up_area' to make it more understandable, 1 means the accident happened in a built-up area and 0 means happened elsewhere. 
characteristics['built-up_area'].replace({1:0, 2:1}, inplace=True)
# -

# ### Fixing incoherent format of year variable

characteristics['year'].value_counts()

# The year format is inconsistent. Until 2018, the year was relative to the year 2000, e.g. "5" for 2005. This changed, however, in 2019 which was labeled as 2019.
# We will change the year format to YYYY.

characteristics['year'].replace({5:2005, 6:2006, 7:2007, 8:2008, 9:2009, 10:2010, 11:2011,
                                                         12:2012, 13:2013, 14:2014, 15:2015, 16:2016, 17:2017, 18:2018}, inplace=True)

# #### Check

characteristics['year'].value_counts()

# ### Fix inconsistent time format

# The time format inconsistent, sometimes it is hhmm, and sometimes hh:mm. We will therefore remove any ":" from the column 

#remove ':' from hhmm
characteristics['hhmm'] = characteristics['hhmm'].apply(lambda s: int(str(s).replace(':','')))


# ### Get weekday and weekend feature

characteristics['date'] = pd.to_datetime({'year':characteristics['year'],
                                                                 'month':dict_of_category_dfs['characteristics']['month'],
                                                                 'day':dict_of_category_dfs['characteristics']['day']})

# +
# New variable: weekday, integer from 0 to 6 representing the weekdays from monday to sunday.
characteristics['day_of_week'] = dict_of_category_dfs['characteristics']['date'].apply(lambda x: x.day_of_week)

# New binary variable: is_weekend, 0 for monday to friday and 1 for saturday and sunday
characteristics['is_weekend'] = (dict_of_category_dfs['characteristics']['day_of_week'] > 4).astype('int')
# -

# ### Remove trailing zeroes from Department variable
# The Department codes are followed by a zero for the years 2005--2018, which has no practical use for us. We will therefore eliminate these trailing zeroes.
# Also, since 2019 all the data is saved as strings. We will convert everything to strings, as this is nominal data, we will not make any calculations with it.

dc = 750
str(dc).rstrip('0')
#.lstrip

# +
def department_converter(dep):
    # Takes in a department code as int and returns a string
    # e.g. 750 will be '75' for Paris
    # and 201 will be '2B'
    if dep == 201:
        return '2A'
    elif dep == 202:
        return '2B'
    elif dep>970:
        return str(dep)
    else:
        return str(dep).rstrip('0')

characteristics.loc[(np.less(characteristics['year'],2019)),'department'] = \
    characteristics[(np.less(characteristics['year'],2019))]['department'].apply(department_converter)
# -

# ### Remove leading zeros from department code
# The dataset from 2021 contains leading zeroes for the department codes 1 to 9. These have to be replaced.

characteristics['department'] = characteristics['department'].apply(lambda code: code.lstrip('0'))

# # Vehicles dataset

# ### Translate variable names from French to English

# We will translate the variable names from French to English for better interpretability and name them more clear (using small letters).

vehicles = vehicles.rename(columns = {'Num_Acc' : 'num_acc','id_vehicule' : 'id_veh' , 'num_veh' : 'num_veh' ,
                           'senc' : 'direction' , 'catv' : 'cat_veh', 'obs' : 'obstacle', 'obsm' : 'obstacle_movable' ,
                          'choc' : 'initial_point' , 'manv' : 'principal_maneuver' , 'motor' : 'motor_veh', 'occutc' : 'num_occupants'})
vehicles.columns

# ### Check of the variables with the most missing values

# Variable num_occupants is representing amount of passangers being victims of an accident when they used public transport system. Missing values are caused by not recording value 0 and keeping the cell empty. For this reason we decided to replace the missing values by 0.

vehicles["num_occupants"] = vehicles["num_occupants"].fillna(0)
vehicles['num_occupants'].isna().sum()
vehicles['num_occupants'].value_counts()

# Variables motor_veh and id_veh represents type of the motorisation of the vehicle. There are 85% missing values in this column. Some of the values of this variable dont specificate exact type but are tracked as unspecified, unknown, other. We have decided to drop this variable as it doesnt have any significant influence on the target variable. 

vehicles = vehicles.drop(columns=['motor_veh','id_veh'])

# 8 Variables have <= 1% missing information, so for those it should be fine to set the missing information just tu zero.

vehicles[['num_acc', 'direction', 'cat_veh', 'obstacle', 'obstacle_movable', 'initial_point', 'principal_maneuver']] = vehicles[['num_acc', 'direction', 'cat_veh', 'obstacle', 'obstacle_movable', 'initial_point', 'principal_maneuver']].fillna(0)
vehicles.isna().sum()

# # Merge all datasets

# ## Compute the percentage of missing data

outer_df = characteristics.merge(right=places, how='outer').merge(users, how='outer').merge(vehicles, how='outer')

print(f'number of rows:........{outer_df.shape[0]}')
print(f'number of variables:...{outer_df.shape[1]}')
na_percentage(outer_df)

# ## Left Join for further investigations
# We will continue working with the left join of the data, as the missing lines miss the most important variables anyway.

df = characteristics.merge(right=places, how='left').merge(users, how='left').merge(vehicles, how='left')
print(df.info())
print(na_percentage(df))

# ## Correlation of the feature variables with the target

cm=df.corr()
cm['grav'].sort_values(ascending=False)[1:]

# The list shows the correlation between each variables and the target variable. Note: The decision whether a variable is important or not has to be based on the absolute value of the correlation.

plt.figure(figsize=(14,14));
sns.heatmap(cm, annot=False);

# ## Fixing incoherency of 'secu' Variable
# Safety equipment until 2018 was in 2 variables: existence and use.
#
# From 2019, it is the use with up to 3 possible equipments for the same user
# (especially for motorcyclists whose helmet and gloves are mandatory).
#
# ### secu1
# The character information indicates the presence and use of the safety equipment:
# -1 - No information
# 0 - No equipment
# 1 - Belt
# 2 - Helmet
# 3 - Children device
# 4 - Reflective vest
# 5 - Airbag (2WD/3WD)
# 6 - Gloves (2WD/3WD)
# 7 - Gloves + Airbag (2WD/3WD)
# 8 - Non-determinable
# 9 - Other
#
# ### secu2
# The character information indicates the presence and use of the safety equipment
#
# ### secu3
# The character information indicates the presence and use of safety equipment
#

df['secu'] = df[df['year']==2007]['secu'].astype(int)
df[df['year']==2007]['secu'].value_counts()

# # Visualizations

# ## Datetime

# We suspect different distributions on weekdays in comparison to weekends. Accidents on weekdays will probably occur mostly when people are commuting to work, i.d. before 09:00 and after 17:00, while on the weekends, people might cause accidents when they return from parties late at night.
# We will plot the relative distribution of accidents in two histograms in one figure.
# The proportion is relative to the total number of accidents in each category, i.e. 1 different for weekends and weekends.

day_time_ticks = (0,300,600,900,1200,1500,1800,2100,2400)
day_time_tick_labels = ('0:00', '03:00','06:00','09:00','12:00','15:00',
                   '18:00','21:00','24:00')
plot_data = pd.DataFrame({'weekdays': (dict_of_category_dfs['characteristics'][dict_of_category_dfs['characteristics']['is_weekend'] != 1 ])['hhmm'],
                 'weekends': (dict_of_category_dfs['characteristics'][dict_of_category_dfs['characteristics']['is_weekend'] == 1 ])['hhmm']})
fig= plt.figure();
sns.histplot(data=plot_data, stat='proportion', bins=24, binrange=(0,2400), common_norm=False);
plt.xticks(ticks=day_time_ticks, 
           labels=day_time_tick_labels);
plt.xlabel('Time of Day')
plt.xlim((0,2400))
plt.title('Distribution of Accidents by Daytime')

# The plot shows, that the temporal distribution is different on the weekends: On weekends, there are far more accidents between 19:00 and 07:00, while there are more accidents on weekday around 09:00 and 18:00. These differences align very well with our hypothesis. We did not expect the peak on weekends around 18:00, though. 
#
# Possible policy measures could be more public transport offers during these times and more police inspections on the weekends near party locations.

# ## Accidents per capita
# ### Outline
# We will investigate, how the ratio accident / habitant differs between the departments. For this, we will create a new DataFrame of Departments. We will count the accidents per Department and import the population data from INSEE to calculate the ratio. 
#
# ### Hypothesis
# We do not expect that this comparison shows stark contrast between the departments. Departments with dense cities will probably have a higher ratio, though. This plot will show us, if there are any outliers.

# +
# Importing data departments from 2019
# source: https://www.insee.fr/fr/statistiques/6011070?sommaire=6011075
departments_2019_df = pd.read_csv('../Data/additional_data/donnees_departements.csv', index_col=2, sep=';')

# Remove leading zeroes from department codes
departments_2019_df.index = departments_2019_df.index.map(lambda idx: idx.lstrip('0'))

# number of accidents by department
n_accidents_by_department = characteristics[characteristics['year']==2019]['department'].value_counts()

# Joining the data
departments_2019_df = departments_2019_df.join(n_accidents_by_department, how='outer') 

# Data is missing for some small departments, we will drop these departments
departments_2019_df.rename(columns={'department':'n_accidents'}, inplace=True)

# Calculate the relative number of accidents per department per 10_000 habitants
departments_2019_df['n_accidents_per_10k'] = departments_2019_df.apply(lambda row: row['n_accidents']*10_000/row['PTOT'] ,axis=1)
departments_2019_df.dropna(axis=0,subset='n_accidents_per_10k', inplace=True)


# +
#departments_2019_df['n_accidents_per_10k'].plot(kind='bar')
departments_2019_df.sort_values(by='n_accidents_per_10k').tail(10).plot.barh(x='DEP',y='n_accidents_per_10k',    
    figsize=(5,5), grid=False, title='Number of Accidents per 10,000 habitants (2019)', legend=False);

departments_2019_df.sort_values(by='n_accidents_per_10k').head(10).plot.barh(x='DEP',y='n_accidents_per_10k',    
    figsize=(5,5), grid=False, title='Number of Accidents per 10,000 habitants (2019)', legend=False);
# -

plt.plot(departments_2019_df['PTOT'], departments_2019_df['n_accidents'], 'x');
plt.title('Accidents in a Department in Function of its Population 2009');
plt.xlabel('Total Population of the Department');
plt.ylabel('Number of Accidents in the Department');


# ### Conclusion
# The differences between the departments are generally higher than expected. Other than the departments without data (which have been dropped before plotting), there are no outliers. The relation between habitants and accidents does not seem to be linear, a quadratic function could be used for fitting here.  

# # Checking Accidents per month and per year

# Hypothesis: In general accidents should be uniform accross all months of the year, and generally accidents should be decreasing across the years especially in covid area since we had a lower volume of car movement across the world

sns.countplot(y = "month" , data = characteristics)
plt.xlabel('Total Number of Accidents');
plt.ylabel('Month');
plt.title('Distribution of Accidents by Month');


sns.countplot(y = "year" , data = characteristics)
plt.xlabel('Total Number of Accidents');
plt.ylabel('Year');
plt.title('Distribution of Accidents by Year');

# # Conclusion, We can see that the number of accidents per month is almost uniform.
# For the years we see a decline in the number of accidents per year, maybe for increased security measures or improved laws and roads.
# We can also see that there is a sharp decrease in the number in the year 2020 probably due to Covid and lockdown in France, and a relative increase after that in 2021.

# ### WE HAVE SEVERAL HEATMAPS MUST CHOOSE ONLY ONE 

# showing frequency of each manevuer before car accident
plt.hist(vehicles["sns.countplot(data=users, x='sexe');
plt.xticks(ticks=[0,1,2],labels=['data missing','male', 'female']);"])
plt.show()

users.sexe.replace(to_replace=-1,value=1,inplace=True)
users.sexe.value_counts()

# Hypothesis: The number of accidents across genders should be equal across males and females.

sns.countplot(data=users, x='sexe');
plt.xticks(ticks=[0,1],labels=['Male', 'Female'])
plt.xlabel('Sex');
plt.ylabel('Total Number of Accidents');
plt.title('Distribution of Accidents by Gender');

# We see that the amount of Males doing accidents is almost double that of females, probably because the amount of males who generally drive are higher than females, or because males are reckless drivers.

users.grav.replace(to_replace=-1,value=1,inplace=True)
users.grav.value_counts()

# We nee to check the severity (gravity) of accidents and its effects on the dirvers, which is our target variable.
# Hypothesis: Only a low number of accidents should result in serious injury or death due to the advanced security systems and road designs.

sns.countplot(data=users, x='grav');  
plt.xticks(ticks=[0,1,2,3], labels=['1\nUnscathed', '2\nKilled',
    '3\nHospitalized\nwounded','4\nLight injury'])
plt.xlabel('gravity');
plt.ylabel('Total Number of Accidents');
plt.title('Number of Accidents according to their gravity');

# Conclusion: We can see that almost 20% of people are Hospitalized and only a very small amount is killed,and hence we can deduce that the target variable is unbalanced.

fig, S = plt.subplots(figsize=(18,18));
sns.heatmap(users.corr() , annot = True );

plt.figure(figsize = (10,9));
sns.countplot( y = places.Rd_Cat);
plt.title('Road Categories with most Accidents');
plt.yticks(ticks=list(range(0,9)),labels=['0=Nans','1 = Highway', '2 = National Road', '3 = Departmental Road', '4 = Communal Way' ,'5 = Off puplic Network','6 = Parking Lot (puplic)' , '7 = Urban Metropolis Roads' , '9 = other']);

# ### Conclusion for road categories with most accidents: 
# Most accidents seem to occur in urban areas. Reasons for this can be oncoming traffic, other road users such as cyclists, narrow or dirty lanes.

g = sns.FacetGrid(places, col = 'Traf_Direct');
g.map(plt.hist, 'Rd_Cat');
g.fig.subplots_adjust(top=0.8);
g.fig.suptitle('Accidents according to traffic direction and road category');


# Legend:
# - Rd.Cats:-----------------------------                    - Traff.Direct: 
# - 1=Highway--------------------------                  - 1=False
# - 2=National Road--------------------              - 0=False 
# - 3=Departmental Road--------------          - 1=One Way 
# - 4=Communal Way------------------               - 2=Bidirectional
# - 5=Off puplic Network---------------         - 3=Separated Carriageways 
# - 6=Parking Lot (puplic)--------------       - 4=With variable assignment Channels
# - 7=Urban Metropolis Roads
# - 9=other                                            

# ### Conclusion for accidents according to traffic direction and road category:
# We can see that with road categories 3 and 4, on which most accidents happen, we have most accidents in places with bidirectional traffic.

plt.figure(figsize = (10,9));
sns.countplot( y = places.Rd_Cond);
plt.title('Road Conditions with most Accidents');
plt.yticks(ticks=list(range(0,11)),labels=['-1=Failure','0=Nans','1 = Normal', '2 = Wet', '3 = Puddles', '4 = Flooded' ,'5 = Snow-Convered','6 = Mud' , '7 = Icy' , '8=Greasy (Oil)', '9 = other']);

# ### Conclusion for road conditions with most accidents:
# By far the most accidents happend during normal weather conditions.

plt.figure(figsize = (10,9));
sns.countplot( y = places.Pos_Acc);
plt.title('Accident Location');
plt.yticks(ticks=list(range(0,9)),labels=['-1=Failure','0=Nans','1 = On Carriageway', '2 = On Emergancy Lane', '3 = On Hard Shoulder', '4 = On Pavement' ,'5 = On Cycle Path / Lane','6 = On Special Lane' , '8=Other']

# ### Conclusion for accident location:
# By far the most accidents happend on the carriage way.

# ## Vehicles dataset visualisations

plt.figure(figsize = (5,5));
sns.countplot( y = vehicles.obstacle_movable);
plt.title('Crashed obstacle');
plt.yticks(ticks=list(range(0,8)),labels=['-1=Nans','0=Nothing','1 = Pedestrian', '2 = Vehicle', '3 = Rail vehicle', '4 = Pet' ,'5 = Wild animal','6 = Other']);

# ### Conclusion for obstacle crashed during accident:
# Most crashed object during car accidents were other vehicles. Followed by no obstacle crashed and crashed pedestrians.

# ### Heat map of Vehicles dataset

# Displaying dataframe correlations as a heatmap 
# with diverging colourmap as RdYlGn
sns.heatmap(vehicles.corr(), cmap ='RdYlGn', linewidths = 0.30, annot = True);

# ### Conclusion for a heat map of vehicles
# We do not observe any strong correlation between the variables in the dataframe Vehicles itself. Negatively correlated are the variables obstacle and obstacle_movable, whereas num_acc and direction are strongly positive correlated - it has no importance on our data as num_acc is only unique identificator of an accident.

# ### Visualisation of most important features

fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(vehicles.corr()[['secu']].sort_values('secu').tail(10),
 vmax=1, vmin=-1, annot=True, ax=ax);
ax.invert_yaxis()

# ### Summary of the most important variables of the data set places.
# Many changes have been made to this data set over time. Unfortunately, no new columns were created for this but existing columns were used for other inputs, so that one column can have several meanings. Unfortunately, some very interesting data cannot be used very well. In general, there is hardly any meaningful connection between the variables. It's not clear which place-descriptive variables give clear clues, but I'll try to come to a conclusion anyway.

# +
plt.figure(figsize = (10,8));

places_a = places.loc[places['Rd_Cond'] == 1]
places_b = places.loc[places['Pos_Acc'] == 1]
places_c = places.loc[places['Rd_Prof'] == 1]
places_d = places.loc[places['Rd_Cat'] == 4]

places_a.loc[places_a["Rd_Cond"]==1, "Rd_Cond"] = "Normal Condition"
places_b.loc[places_b["Pos_Acc"]==1, "Pos_Acc"] = "On the Road"
places_c.loc[places_c["Rd_Prof"]==1, "Rd_Prof"] = "Flat Road"
places_d.loc[places_d["Rd_Cat"]==4, "Rd_Cat"] = "Communal Way"

plt.hist([places_b.Pos_Acc, places_c.Rd_Prof, places_a.Rd_Cond,places_d.Rd_Cat],color=['blue','lightgrey','red','green'], width = 0.7, rwidth=50);
plt.ylabel('Count');
plt.title('The Most Common Accidents');
# -

# ### Conclusion for the most common accidents:
# I'm not saying that all of these accidents are related, but I think one can deduce thatfrom the total number of accidents (1.121.571 accidents) a high number of accidents occur with "good" parameters. At least as far as the place of the accident is concerned.
