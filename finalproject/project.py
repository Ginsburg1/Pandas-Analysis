import pandas as pd
import numpy as np
from pandas import Series
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

# import trash wheel data
stdMTW = pd.read_excel('finalproject/TrashWheelCollectionData.xlsx',
                      skiprows=(0,1), sheet_name='Mr. Trash Wheel',
                      names=['Dumpster',
                                'Month',
                                'Year',
                                'Date',
                                'Weight (tons)', 
                                'Volume (cubic yards)', 
                                'Plastic Bottles', 
                                'Polystyrene', 
                                'Cigarette Butts', 
                                'Glass Bottles', 
                                'Plastic Bags', 
                                'Wrappers', 
                                'Sports Balls', 
                                'Homes Powered*'],
                      usecols='A:N')

# import churn modeling data
stdCM = pd.read_excel('finalproject/Churn_Modelling.xlsx',
                      skiprows=(0,1), sheet_name='Churn_Modelling',
                      names=['RowNumber',
                             'CustomerID',
                             'Surname',
                             'CreditScore',
                             'Geography',
                             'Gender',
                             'Age',
                             'Tenure',
                             'Balance',
                             'NumOfProducts',
                             'HasCrCard',
                             'IsActiveMember',
                             'EstimatedSalary',
                             'Exited'],
                      usecols='A:N')


# import candy sales great chefs data
stdCSGC = pd.read_excel('finalproject/CandySalesGreatChefs.xlsx',
                      skiprows=(0,1), sheet_name='Great Chefs',
                      names=['OrderDate',
                             'ProductID',
                             'Product Name',
                             'Quantity Sold',
                             'Unit Sales Price',
                             'Total Sales Revenue',
                             'Unit Cost',
                             'Total Cost',
                             'Unit Profit',
                             'Total Profit',
                             'CustomerID',
                             'Customer',
                             'SalesRep'],
                      usecols='A:M')



# Part 1 A: pivot table
print(stdCM.pivot_table(index=['Geography'], values=['CreditScore','EstimatedSalary'], aggfunc=np.average))
stdCM.pivot_table(index=['Geography'], values=['CreditScore','EstimatedSalary'], aggfunc=np.average).plot(kind='bar',alpha=0.75, rot=0)
plt.xlabel('Region')
plt.ylabel('Average')
plt.title('Geography vs. Average')
plt.legend(loc='upper right')
plt.savefig('finalproject/1A.png', bbox_inches='tight')
plt.show()



# Part 1 B: crosstab
print(pd.crosstab(stdCSGC.ProductID, stdCSGC.Customer, margins=True, margins_name="Total"))
pd.crosstab(stdCSGC.ProductID, stdCSGC.Customer, margins=True, margins_name="Total").plot(kind='bar',alpha=0.75, rot=0)
plt.gcf().subplots_adjust(bottom=0.15)
plt.xlabel('Product ID')
plt.ylabel('Customer Count')
plt.title('Product ID vs. Customer')
plt.savefig('finalproject/1B.png', bbox_inches='tight')
plt.show()


# Part 2: bin the data
Bins = [0, 500, 1000, 2000, 4000, 6000, 10000]
GradeBins = pd.cut (stdMTW.Wrappers, Bins)
GradeCounts = pd.value_counts(GradeBins) 

# read in specific data
GradeCounts.sort_index(inplace=True)
GradeCounts.plot(kind='bar',alpha =0.75, rot=0)
plt.xlabel('Wrappers')
plt.ylabel('Frequency')
plt.title('Wrappers Frequency Distribution')
plt.savefig('finalproject/2.png', bbox_inches='tight')
plt.show()

# Part 3: more grpahs (pie chart)
ClassCnts = pd.value_counts (stdCM.Geography)
ClassCnts.plot (kind='pie')
plt.title('Geography Frequency Distribution')
plt.legend(loc='upper right')
plt.savefig('finalproject/3A.png', bbox_inches='tight')
plt.show()

# 3b: bar graph
ClassCnts2 = pd.value_counts (stdCSGC.SalesRep)
ClassCnts2.plot (kind='bar',alpha=0.75, rot=0)
plt.ylabel('Frequency')
plt.xlabel('Sales Rep')
plt.title('Sales Rep Frequency Distribution')
plt.savefig('finalproject/3B.png', bbox_inches='tight')
plt.show()

# 3c: scatter plot
plt.scatter(stdMTW['Weight (tons)'], stdMTW['Volume (cubic yards)'])
plt.xlabel('Weight (tons)')
plt.ylabel('Volume (cubic yards)')
plt.title('Weight (tons) vs. Volume (cubic yards)')
plt.savefig('finalproject/3C.png', bbox_inches='tight')
plt.show()


# Part 4: 1 sample t-test
alpha = .05
hypothesis = 38.73

tmean, p_valmean = stats.ttest_1samp(stdCM['Age'], hypothesis)
print(p_valmean)
if p_valmean < alpha :
    tmeanHo=False
    ttype='reject the Null Hypothesis, the mean is different from 38.73'
else:
    tmeanHo=True
    ttype='fail to reject the Null Hypothesis, the mean is simular to 38.73'

plt.hist(stdCM['Age'])
plt.axvline(stdCM['Age'].mean(), color='r', linestyle='solid', linewidth=2, label="Mean")
plt.axvline(hypothesis, color='b', linestyle='solid', linewidth=2, label="Hypothesized Mean")
plt.legend(loc='upper right')
plt.title(f't: {round(tmean,3)}, p-val: {round(p_valmean,4)}', size=10)
plt.suptitle(ttype, size=10)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('finalproject/4.png', bbox_inches='tight')
plt.show()



# Part 5: 2 sample t-test
Female = stdCM[stdCM['Gender'] == 'Male']
Male = stdCM[stdCM['Gender'] == 'Female']

alpha = .05
tvar, p_valvar = stats.bartlett(Female['CreditScore'], Male['CreditScore'])
if p_valvar < alpha:
    tEqVar=False
    ttype='There is no significant difference in the variances of CreditScore and Age in the population.'
else:
    tEqVar=True
    ttype='There is a significant difference in the variances of CreditScore and Age in the population.'

alpha = .05
tmean, p_valmean = stats.ttest_ind(Female['CreditScore'], 
                                   Male['CreditScore'],equal_var=tEqVar)
y=[Female['CreditScore'], Male['CreditScore']]
plt.boxplot(y)
plt.title(f't: {round(tmean,3)}, p-val: {round(p_valmean,4)}', size=10)
plt.suptitle(ttype, size=10)
plt.xticks(range(1,3),[f"Female: {round(Female['CreditScore'].mean(),2)}",
f"Male: {round(Male['CreditScore'].mean(),2)}"])
plt.ylabel("Credit Score")
plt.savefig('finalproject/5.png', bbox_inches='tight')
plt.show()

# Part 6: anova test
alpha = .05
tvar, p_valvar = stats.bartlett(stdCM['CreditScore'], stdCM['Age'])

if p_valvar < alpha:
    tEqVar=False
    ttype='The variances are not equal'
else:
    tEqVar=True
    ttype='The variances are equal'

# makegraph
y=[stdCM['CreditScore'], stdCM['Age']]
plt.boxplot(y)
plt.title(f't: {round(tmean,3)}, p-val: {round(p_valmean,4)}', size=10)
plt.suptitle(ttype, size=10)
plt.xticks(range(1,3),[f"Credit Score: {round(stdCM['CreditScore'].mean(),2)}",
f"Age: {round(stdCM['Age'].mean(),2)}"])
plt.ylabel("Credit Score")
plt.savefig('finalproject/6.png', bbox_inches='tight')
plt.show()


# Part 7: linear regression
xvar='Age'
yvar='Balance'
x=stdCM[xvar]
y=stdCM[yvar]
slope, intercept, r_value, p_val, std_err = stats.linregress(y=y,x=x)

if np.sign(slope) < 0:   
    slsign = ""
else:
    slsign = "+"

regeq = f"{yvar} = {round(intercept,3)}{slsign}{round(slope,3)} {xvar}"
print(f"The equation is {regeq}")

alpha=.05
if p_val < alpha:
    print("Conclusion: Reject Ho: X does help predict Y")
else:
    print("Conclusion: Fail to Reject Ho: We can't reject that X doesn't help to predict Y")


plt.scatter(x,y,color='black')
xyCorr = round(x.corr(y),3)
plt.suptitle(f"Correlation: {xyCorr}  R-Squared: {round(r_value**2,4)} p-value: {round(p_val,4)}")
plt.title(regeq, size=10)
predict_y = intercept + slope * x
plt.plot(x,predict_y, 'r-')
plt.xlabel(xvar)
plt.ylabel(yvar)
plt.savefig('finalproject/7.png', bbox_inches='tight')
plt.show()

def MultipleRegressionAnalysis(MRModel) :
    r2adj = round(MRModel.rsquared_adj,2)                 
    p_val = round(MRModel.f_pvalue,4)
    
    coefs = MRModel.params
    coefsindex = coefs.index
    regeq = round(coefs[0],3)                              
    cnt = 1
    for i in coefs[1:]:
        regeq=f"{regeq} + {round(i,3)} {coefsindex[cnt]}"
        cnt = cnt + 1

    print("Adjusted R-Squared: " + str(r2adj))
    print("P value: " +str(p_val))
    if p_val < alpha :
        print ("Reject Ho: X variables do predict Loss")
    else :
        print ("Do not reject Ho")
    print(regeq)

def my_multreg(model, ydata):
    r2adj = round(model.rsquared_adj,2) #use for multiple regression
    p_val = round(model.f_pvalue,4)
    coefs = model.params
    coefsindex = coefs.index 
    regeq = round(coefs[0],3)
    cnt = 1
    for i in coefs[1:]:
        regeq=f"{regeq} + {round(i,3)} {coefsindex[cnt]}"
        cnt = cnt + 1
    #Scatterplot for Multiple Regression - y vs predicted y
    miny=ydata.min()
    maxy=ydata.max()
    predict_y = model.predict()
    plt.scatter(ydata,predict_y)
    diag = np.arange(miny,maxy,(maxy-miny)/50)
    plt.scatter(diag,diag,color='red',label='perfect prediction')
    plt.suptitle(regeq)
    plt.title(f' with adjR2: {r2adj}, F p-val {p_val}',size=10)
    plt.xlabel(ydata.name)
    plt.ylabel('Predicted ' + ydata.name)
    plt.legend(loc='best')
    plt.savefig('finalproject/8A.png', bbox_inches='tight')
    plt.show()
    #Scatterplot residuals 'errors' vs predicted values 
    resid = model.resid
    predict_y = model.predict()
    plt.scatter(predict_y, resid)
    plt.suptitle(regeq)
    plt.hlines(0,miny,maxy)
    plt.ylabel('Residuals')
    plt.xlabel('Predicted ' + ydata.name)
    plt.savefig('finalproject/8B.png', bbox_inches='tight')
    plt.show()


# Part 8: multiple linear regression
alpha = .05
DCmodel = ols("CreditScore ~ Tenure + Age",data=stdCM).fit()
print("ANALSIS FOR DETERMINING IF CreditScore AND Tenure and age are related")
MultipleRegressionAnalysis(DCmodel)
ydata=stdCM['CreditScore']
my_multreg(DCmodel,ydata)



