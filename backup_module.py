#!/usr/bin/env python
# coding: utf-8

# # 단순회귀의 이해

# In[1]:


#분석 준비
##수치 분석에 사용하는 라이브러리
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

#그래프를 그리기 위한 라이브러리
import matplotlib as plt
import seaborn as sns
sns.set()

#선형모델을 추정하는 라이브러리
import statsmodels.formula.api as smf
import statsmodels.api as sm

#표시 자릿수 지정
get_ipython().run_line_magic('precision', '3')

#그래프를 주피터 노트북에 그리기 위한 설정
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#데이터 읽어오기. (사이버캠퍼스에서 이번주 데이터 다운받아서 폴더에 넣어주세요.)
beer = pd.read_csv("5-1-1-beer.csv")
print(beer.head())


# In[3]:


#데이터간 관계를 알기 위해 먼저 plot으로 관련성을 살펴보시죠. 늘 습관을 들이시면 좋습니다.
##시각화에서 배우셨던 seaborn에서 jointplot을 써보겠습니다.
sns.jointplot(x = "temperature", y = "beer",
             data = beer, color = 'black')


# In[4]:


#그래프를 보시면 가로축이 "기온", 세로축이 "매상"인데요. 가로축과 매상이 어느정도 함께 움직이는 것처럼 보이죠?
##이 관계를 바탕으로 맥주 매상을 설명할 수 있는 모델을 만들어보려고 합니다.
###종속변수를 맥주 매상으로, 독립변수를 기온으로 한 단순회귀모델 입니다.
####통계모델을 추정하기 위해서 statsmodel을 이용해보시죠.


# In[5]:


# stasmodel에서도 ols (Ordinar Least Squares 범용최소제곱법)을 써보시죠.
## ols는 교과서에서도 나오지만 모집단이 정규분포임을 가정했을때 최대우도법의 결과는 최소제곱법 결과와 일치합니다.
##우도 설명(https://everyday-tech.tistory.com/entry/%EC%B5%9C%EB%8C%80-%EC%9A%B0%EB%8F%84-%EC%B6%94%EC%A0%95%EB%B2%95Maximum-Likelihood-Estimation
### 모델의 구조를 지정하는 것이 formula 입니다. "beer ~ temparature"는 "종속변수가 beer, 독립변수가 temparature"임을 의미합니다.
### fit()을 통해서 파라미터를 자동으로 추정해줄 것입니다.
lm_model = smf.ols(formula = "beer ~ temperature", data=beer).fit()


# In[6]:


#summary 함수를 통해서 결과를 추정해볼까요?
lm_model.summary()


# In[7]:


#많은 내용이 있지만 우리가 관심있게 봐야 할 부분은 B0과 B1에 해당하는 intercept와 temparature 입니다.
##coef(=coefficient 계수)라고 적혀있는 곳이 (계수값)이고요. std가 표준오차, t값, 귀무가설에 대한 p값.
###즉, p값에 따라 우리가 주장하고자 하는 가설이 받아들여지는지 여부를 결정할 수 있는데요. 
#### 여기서는 "기온에 따라 매상이 변화가 있는가?"를 귀무가설로 놓고,기각되었으므로 "기온에 따라 매상에 변화가 있다"는 대립가설이 받아들여집니다.
##### 기온이 1도 올라갈때, 매상이 얼마나 올라갔는지는 여기의 "tempature - coef값이며 0.7654라고 할 수 있습니다"


# ## 회귀직선 그리기

# In[8]:


#회귀직선은 모델에 의한 종속변수의 추측값을 직선으로 표시한 것입니다.
##Seaborn을 써서 회귀직선 그래프를 그려보겠습니다 (lmplot).
sns.lmplot(x = "temperature", y = "beer", data = beer, 
           scatter_kws = {"color":"black"},
           line_kws = {"color":"black"})


# ## 결정계수 이해하기

# In[9]:


#summary 함수의 출력에 있는 R-suqred가 결정계수이고, 데이터를 모델에 적용했을 때의 적합도를 평가한 지표
##모델의 설명력이라고 부릅니다.
###수정된 결정계수는 독립변수 숫자가 늘어나는 것에 대해 패널티를 적용한 것이며, 결정계수가 높아지면 과학습이 일으킬 수 있으므로 조정해야 합니다.


# In[10]:


lm_model.rsquared


# In[11]:


lm_model.rsquared_adj


# ## 잔차 그래프 그리기

# In[12]:


#잔차의 특징을 보는 가장 간단한 방법이 바로 잔차의 히스토그램을 보는 것입니다.
##히스토그램을 보고 정규분포의 특징을 갖고 있는지 살펴보시죠.
###먼저 잔차를 계산하는건 파이썬 함수를 쓰시죠.
resid = lm_model.resid
sns.distplot(resid, color = 'black')


# ## Q-Q 플롯

# In[13]:


#이론상의 분위점과 실제 데이터의 분위점을 산포도 그래프로 그린 것을 Q-Q플롯이라고 합니다 (Q=Quantile)
##Q-Q플롯을 통해 이론상의 분위점과 실제 데이터의 분위점을 구해서 둘을 비교하는 것으로 잔차가 정규분포에 접근해있는지 시각적으로 판단할 수 있습니다.
###Q-Q플롯은 sm.qqplot 함수를 이용해서 그릴 수 있습니다.
fig = sm.qqplot(resid, line = "s")


# # 다중회귀분석 이해

# In[15]:


#이번에는 독립변수가 여러 개인 모델을 살펴보시죠.
## 온라인 캠퍼스에서 '5-3-1-lm-model.csv'파일을 다운로드 하시죠.
### 그리고 헤드 함수를 써서 어떤 변수들이 어떤 형식으로 있나 보실까요?
sales = pd.read_csv("5-3-1-lm-model.csv")
print(sales.head())


# #### 변수로는 습도(humidity), 가격(price), 판매량(sales), 기온(temperature), 날씨(weather)가 있네요.

# #### 앞서도 말씀드린대로 변수가 주어지면 가장 먼저하실 것이 데이터를 이용해서 그래프를 그려보는 것입니다.

# In[16]:


sns.pairplot(data = sales, hue = "weather", palette = "gray")


# #### 근데 결과를 보니 3행의 sales 데이터가 습도, 가격과 관련이 있는지 알기 어렵죠.

# In[19]:


#그러므로 이번에는 독립변수가 4개 다 들어간 모델로 추정을 해보시죠.
lm_sales = smf.ols("sales ~ weather + humidity + temperature + price", data = sales).fit()
#추정된 결과
lm_sales.params


# In[20]:


#분석결과 흥미로운 것은 price가 -(마이너스)라는 점인데요. 가격이 오르면 매상이 떨어진다는 것입니다.
##그러면 실제로 가격이 오르면 매상이 떨어지는지, 다른 변수도 그러한지 확인해보시죠.
lm_sales.summary().tables[1]


# In[24]:


#결과를 보시면 humidity는 p값이 0.05보다 크므로 유효한 변수로 보기 어렵죠.
##그러므로 우리가 할 수 있는 결과 해석은 날씨, 기온과 가격이 매상에 통계적으로 유의한 영향을 미친다는 것입니다.
### 자 그러면 통계적으로 유의하지 않은 변수를 제외한 모델을 다시 만들어서 계수를 추정해보시죠.
lm_sales2 = smf.ols("sales ~ weather + temperature + price", data = sales).fit()
lm_sales2.summary().tables[1]


# In[26]:


#습도를 제외하고 계수가 바뀌었죠. 이처럼 종속변수에 영향을 미치는지를 알기 위해서는 
## 어떤 독립변수가 "유의미한"영향을 미치는지 알아야 합니다.


# ## Assumption(가정) 체크하기

# In[28]:


#회귀분석을 실행할때 우리가 다뤄야 할 중요한 assumption이 있습니다.
#보통 네 가지 assumption을 확인하게끔 하는데요 "선형", 다중공선성", "잔차의 정규분포성","분포의 등분산"입니다.
##맨하탄 집값 데이터로 이를 살펴보시죠.


# In[30]:


#우선 데이터를 사이버캠퍼스에서 다운로드 받아주세요 (mahattan.csv)
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
import statsmodels.api as sm
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# In[31]:


#https://towardsdatascience.com/predicting-manhattan-rent-with-linear-regression-27766041d2d9
df = pd.read_csv('manhattan.csv')


# In[32]:


df.head()


# In[33]:


df.info()


# In[34]:


df.describe()


# In[36]:


#borough랑 rental_id 필요없는 정보 없애보시죠. drop메서드를 쓰며, axis=1은 칼럼의 레이블을 의미하고, axis=-은 인덱서를 의미합니다.
df.drop('borough', axis=1, inplace=True)
df.drop('rental_id', axis=1, inplace=True)


# In[37]:


#저희 결측값 확인하는 방법이 뭐가 있었나요? df.isna() 써보시죠.
df.isna().sum()


# In[38]:


df.neighborhood.unique()


# In[39]:


df.rent.describe()


# In[40]:


#figsize는 inch이며, 디폴트는 (6.4,4.8)이고, pixel로 바꾸는 것은 https://sosomemo.tistory.com/62 참고하세요.

df.hist(grid=False, figsize=(15,15), layout=(5,3), color='#6FD1DF');

#distribution을 보시죠.
#rent, bathroom, building age, floors가 오른쪽으로 skewed되어 있죠.


# In[41]:


keys = list(df.neighborhood.value_counts().keys())
vals = list(df.neighborhood.value_counts())


# In[42]:


#명목형 변수인 neighborhood의 숫자를 보기 위해서 horizontal수평적 bar plot을 그려보시죠.
## 데이터 세트의 많은 집들이 upper west와 upper east side neighborhood에 있는걸 알 수 있습니다.
sns.set_theme(style="darkgrid")

f, ax = plt.subplots(figsize=(12, 8))

ax.set_title('Total Rental Units by Neighborhood', 
             fontname='silom', fontsize=15)

ax.set_xlabel('Count', 
             fontname='silom', fontsize=12)

ax.set_ylabel('Neighborhood', 
             fontname='silom', fontsize=12)

sns.barplot(x=vals, y=keys,
            color="rebeccapurple");


# In[43]:


avg_rents = df.groupby('neighborhood').mean().reset_index()[['neighborhood', 'rent']]
avg_rents['rent'] = [round(r) for r in avg_rents.rent] 
avg_rents.sort_values(by=['rent'], ascending=False, inplace=True)


# In[45]:


#추가적으로 궁금한 것은 어떤 neighborhood에서 가장 평균적으로 높고 낮은 rent를 갖고 있는지 보시죠.
##neighborhood를 그룹별로 나누고 계산을 해서 box plot으로 그렸습니다.
order = list(avg_rents.neighborhood)
plt.figure(figsize=((17, 10)))
ax = sns.boxplot(x="neighborhood", y="rent",
                 data=df, linewidth=2.5, 
                 order=order)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.xlabel("Neigborhood", size=15, fontname='silom')
plt.ylabel("Rent", size=15, fontname='silom')
plt.title("Manhattan Rental Costs by Neighborhood", size=18, fontname='silom');

#Soho, Tribeca, Central Park South가 가장 높은 평균치를 가지고 있죠. Littel Italy, Inwood Manhattanvile이 낮죠. 


# ## 본격적으로 assumption을 확인해보시죠

# In[46]:


# Linearity(선형)
neighborhood_dummies = pd.get_dummies(df.neighborhood).drop('Manhattanville', axis=1)
df = pd.concat([df, neighborhood_dummies], axis=1).drop('neighborhood', axis=1)


# In[48]:


scatter_matrix (df, figsize = (40,40), alpha = 0.9, diagonal = "kde", marker = "o");


# In[49]:


df_correlated = df[['rent', 'bedrooms', 'bathrooms', 'size_sqft']]
scatter_matrix(df_correlated, figsize = (16,10),
               alpha = 0.9, diagonal = "kde", marker = "o")

plt.suptitle('Scatter Matrix of Features Correlated with Rent', 
          fontsize=16,
          fontname='silom')
("")

#suqare footage 평방 피트가 올라갈수록 rental cost로 선형으로 올라가는 것을 발견함.


# # 다중공선성

# In[50]:


corr = df_correlated.corr()

plt.figure(figsize=(16,10))

plt.title('Heatmap of Features Correlated with Rent', 
          fontsize=16,
          fontname='silom'
)

ax = sns.heatmap(
    corr, 
    cmap='coolwarm',
    center=0, 
    linewidth=1,
    linecolor='lavender',
    annot = True
)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    fontname='silom',    
    horizontalalignment='right')
    
ax.set_yticklabels(
    ax.get_xticklabels(),
    fontname='silom',    
    horizontalalignment='right'
);

#독립변수간의 상관관계가 높을 경우에 다중공선성의 문제가 있음. .7이하일 경우 받아 드리는?
#데이터로 봤을때, 다중공선성 문제때문에 rent를 예측할때는 단순선형회귀에서 평방피트로만 예측해야 하는 문제로 보임.
#왜냐하면 너무 상관이 높음.


# ## 잔차의 정규성

# In[51]:


#잔차의 정규성을 확인하는데, 실측치와 회귀 모형의 예측치 차이를 잔차(residual)이라고 하고, 
## 이 잔차가 정규성을 띄고 있는지 역시 확인해야 합니다.-잔차가 정규 분포가 되도록 회귀 모형을 만들어야 모형이 데이터에 잘 적합한 것임.
y = df.rent
x = np.array(df['size_sqft']).reshape(-1, 1)
model = LinearRegression()
model.fit(x, y)


# In[52]:


residuals = y - model.predict(x)


# In[55]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))
residuals = y - model.predict(x)
sns.histplot(residuals, ax=ax1, kde=True)
ax1.set(xlabel="Residuals")
sm.qqplot(residuals, stats.t, distargs=(4,), fit=True, line="45", ax=ax2);
plt.suptitle('Linear Model Residuals', fontname='silom', fontsize=15);


# ## 등분산 확인

# In[54]:


#마지막으로 집단간 분산이 같은지를 확인하는 것임.
fig = plt.figure(figsize=(15,7))
plt.scatter(model.predict(x), residuals, color='rebeccapurple')
plt.axhline(y=0, color='lightgreen', linestyle='-', linewidth=4)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Fitted Vs. Residuals', fontname='silom', fontsize=15)
ax = plt.gca()
ax.set_facecolor('#f9f6fc')
plt.show()
plt.show()


# ## 모델 적합도를 평가해보자.

# In[56]:


def predict_rent(sqft):
    return model.predict(np.array([sqft]).reshape(1, -1))


# In[57]:


predict_rent(250)


# In[58]:


predict_rent(500)


# In[59]:


1223*2


# In[60]:


print('R-Squared:', round(model.score(x, y), 2))
print('Coefficient:', round(model.coef_[0], 2))
print('Intercept:', round(model.intercept_, 2))


# In[ ]:




