import pandas as pd 
import matplotlib.pyplot as plt
df = pd.read_csv('train.csv')
#print(df.info())
#print(df['followers_count'].min())

no_name = 0
no_name_yes = 0
so_so = 0
so_so_yes = 0
star = 0
star_yes = 0


df.drop(['id','career_end', 'career_start' , 'occupation_name', 'occupation_type', 'last_seen', 'city', 'people_main', 'life_main', 'langs', 'education_status', 'education_form', 'bdate'], axis=1, inplace=True)
df.info()

def follower(row):
    global no_name, so_so, star, no_name_yes, so_so_yes, star_yes
    if row['followers_count'] < 1000:
        no_name += 1
        if row['result'] == 1:
            no_name_yes += 1 
            
    elif row['followers_count'] < 5000:
        so_so += 1
        if row['result'] == 1:
            so_so_yes += 1
            
    elif row['followers_count'] > 5000:
        star += 1 
        if row['result'] == 1:
            star_yes += 1



def foll_apply(followers):
    if followers < 1000:
        return 0
    elif followers < 5000:
        return 1
    elif followers > 5000:
        return 2

df.apply(follower, axis=1)

df['followers_count'] = df['followers_count'].apply(foll_apply)


no_name_procent = (no_name_yes/no_name) * 100
so_so_procent = (so_so_yes/so_so) * 100
star_procent = (star_yes/star) * 100


s = pd.Series(index = ['не популярные','Средние','звезды'], data = [no_name_procent, so_so_procent, star_procent])
s.plot(kind = 'barh', figsize= (12,5))
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

x = df.drop('result', axis=1)
y = df['result']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print(y_test)
print('правильные исходы:', round(accuracy_score(y_test, y_pred) * 100 ,2))