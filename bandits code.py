# %%
import pandas as pd
import numpy as np
import random 
import datetime

# Choose stocks and clean data
stockdata = pd.read_excel('2005-2020_week.xlsx',index_col=0)
choosestock = ['600180.SH','000400.SZ','000858.SZ','600381.SH','600123.SH','000852.SZ','000666.SZ']
start = datetime.datetime(2013,1,1)
hisstart = datetime.datetime(2010,1,1)
hisdata = stockdata.loc[stockdata.index >= hisstart, choosestock]
curdata = stockdata.loc[stockdata.index >= start, choosestock]
rt = (curdata.pct_change()+1).dropna().apply(np.log)
hisrt = hisdata.pct_change(periods = 150)
hisrt = hisrt[hisrt.index >= start]/150
# %%

# Define a funtion to choose arms on random
def choosearm(p):
    random.seed(1111)
    q = np.random.random(1)
    n = sum(q>np.cumsum(p))
    return n
# %%
# Define Hedge Algorithm (FULL FEEDBACK AND ADVERSARIAL COSTS)

def hedge_alg(obj, epsilon, T=True):
    if T == True:
        T1 = len(obj)
    else: T1 == T
    res1 = pd.DataFrame(columns = obj.columns,index = obj.index)
    # intialize parameters
    num_arms = len(obj.columns)
    cost = np.zeros(num_arms)
    w1 = np.ones(num_arms) 

    # loop for hedge algorithm
    for t in range(0,T1-1):
        pt = w1/sum(w1)
        n = choosearm(pt)
        cost[n] = -obj.iloc[t,n]
        w1 = w1*(1-epsilon)**cost
        res1.iloc[t+1,n] = obj.iloc[t+1,n]
        # return (np.argmax(w1),w1[np.argmax(w1)])
    return res1
hed_res = hedge_alg(rt, 0.2, T=True)
hed = (np.e**pd.DataFrame(hed_res.stack(),columns=['hedge_return'])).cumprod().reset_index().set_index('level_0')
# %%
# Define epsilon-greedy algorithm
def changearm(arms,cur_arm):
    if type(arms) != list:
        arms = arms.tolist()
    arms_copy = arms.copy()
    del arms_copy[cur_arm]
    import random
    random.seed(1234)
    new_arm_name = random.choice(arms_copy)
    new_arm = arms.index(new_arm_name)
    return new_arm


def epsilon_greedy(obj,T = True):
    if T == True:
        T1 = len(obj)
    else: T1 == T
    df = pd.DataFrame(columns=obj.columns,index = obj.index)
    num_arms = len(obj.columns)
    arm = 0
    df.iloc[0,arm] =  obj.iloc[0,arm]
    for t in range(1,T1):
        # epsilon = t**(-1/3)*(num_arms*np.log(t))**(1/3)
        epsilon = 0.2
        random.seed(1234)
        suc = np.random.random(1)
        if suc <= epsilon:
            arm = changearm(obj.columns,cur_arm = arm)
            df.iloc[t,arm] =  obj.iloc[t,arm]
        else:
            arm = np.argmax(df.mean())
            df.iloc[t,arm] =  obj.iloc[t,arm]
    return df
eps_res = epsilon_greedy(rt, T=True)
eps = (np.e**pd.DataFrame(eps_res.stack(),columns=['epsilon_greedy_return'])).cumprod().reset_index().set_index('level_0')

# %%
# UCB1 algorithm

def UCB1(obj,T = True):
    if T == True:
        T1 = len(obj)
    else: T1 == T
    df = pd.DataFrame(columns=obj.columns,index = obj.index)
    df.iloc[0,:] =  obj.iloc[0,:]
    for t in range(1,T1):
        arm = np.argmax(df.mean()+np.sqrt(2*np.log(t)/df.count()))
        df.iloc[t,arm] =  obj.iloc[t,arm]
    return df

ucb_res = UCB1(rt, T=True)
ucb = (np.e**pd.DataFrame(ucb_res.stack(),columns=['ucb1_return'])).cumprod().reset_index().set_index('level_0')

# %%
# Mean average
hisrt_mean = hisdata.pct_change(periods = 8)
hisrt_mean = hisrt_mean[hisrt_mean.index >= start]/8

def meanave(obj):
    df = pd.DataFrame(0,columns=obj.columns,index = obj.index)
    for i in range(len(obj)-1):
        df.iloc[i+1,np.argmax(hisrt_mean.iloc[i,:])] = -1
        df.iloc[i+1,np.argmin(hisrt_mean.iloc[i,:])] = 1
    res = df*obj
    res['mean_average_return'] = res.sum(axis = 1)
    return res['mean_average_return']

mea_res = meanave(rt)
mea = (np.e**pd.DataFrame(mea_res,columns=['mean_average_return'])).cumprod()





# %%
# ES optimal algorithm
def F(u,alpha,t,delta,gamma,obj):
    UH = 0
    UR = 0
    for s in range(t):
        UH = max(-(sum(u*np.array(obj.iloc[s]))+UH+alpha),0)+UH
        UR = max(-(sum(u*np.array(obj.iloc[s]))+alpha),0)+UR
    Fgamma = alpha + 1/((delta+t-1)*(1-gamma)) * (UH+UR)
    return Fgamma

# solve optimal weight

def portfolio_selection(obj,obj_history, epsilon, T, lamb, gamma=0.95):
    w_data = pd.DataFrame(columns=rt.columns, index = rt.index)
    res1 = pd.DataFrame(columns=['stock'],index = rt.index)
    # intialize parameters
    num_arms = len(obj.columns)
    cost = np.zeros(num_arms)
    w1 = np.ones(num_arms) 
    epsilon=np.sqrt(np.log(num_arms)/T)

    from scipy import optimize
    # loop for hedge algorithm
    for t in range(0,T-1):
        pt = w1/sum(w1)
        n = choosearm(pt)
    # Optimize for WC
        def F(u):
            delta = t
            UH = 0
            UR = 0
            alpha = u[num_arms]
            for s in range(t):
                U1 = sum([u[i]*obj_history.iloc[s,i] for i in range(num_arms)])
                U2 = sum([u[i]*obj.iloc[s,i] for i in range(num_arms)])
                UH = max(-(U1+alpha),0)+UH
                UR = max(-(U2+alpha),0)+UR
            Fgamma = alpha + 1/((delta+t-1)*(1-gamma)) * (UH+UR)
            return Fgamma
        def cons(u):
            return sum([u[i] for i in range(len(u)-1)]) - 1
        def ieq_cons(u):
            for i in range(len(u)-1):
                return u[i]
        print(t)
        from scipy.optimize import Bounds
        w_c = optimize.fmin_slsqp(F,[1/num_arms]*(num_arms)+[0],eqcons=[cons,],ieqcons=[ieq_cons,])[:-1]
    # Calculate WM
        cost[n] = 1/(1+np.exp(-obj.iloc[t,n]))
        w1 = w1*(1-epsilon)**cost
        res1.iloc[t+1,0] = stockdata.columns[np.argmax(w1)]
        w_star = lamb
        w = (w_c*(1-lamb)).tolist()
        w[np.argmax(w1)] = w[np.argmax(w1)]+w_star
        w_data.iloc[t+1] = w

    return res1, w_data

res1, w_data = portfolio_selection(rt,hisrt, 0.05, len(rt.index), 0.5, gamma=0.95)


# %%

res = res1.merge(w_data,left_index=True,right_index=True)

# Calculate profit and loss
pandl = w_data*rt

pandl['totalreturn'] = pandl.apply(lambda x: sum(x),axis = 1)

# Delete outliers which is caused by failure of optimization
pandl = pandl[abs(pandl['totalreturn'])<3]

# %%
import matplotlib.pyplot as plt
fig,ax = plt.subplots(3,1,figsize=(12,16))
ax[0].plot(np.cumprod(np.array(np.e**pandl['totalreturn'])))
ax[0].set_title('Combined Portfolio')
ax[0].set_ylabel('Cumulative Wealth')

(np.e**rt).cumprod().plot(ax = ax[1])
ax[1].legend(loc='best')
ax[1].set_ylabel('Relative Price')
ax[1].set_title('Performance of Singal Stock')

hed['hedge_return'].plot(ax = ax[2],label = 'hedge algorithm')
eps['epsilon_greedy_return'].plot(ax = ax[2],label = 'epsilon greedy algorithm')
ucb['ucb1_return'].plot(ax = ax[2],label = 'UCB1 algorithm')
mea.plot(ax = ax[2],label = 'Mean Average')
ax[2].legend(loc='best')
ax[2].set_xlabel('Time')
ax[2].set_ylabel('Singal Method')
ax[2].set_title('Performance of Singal Algorithm')
for tick in ax[2].get_xticklabels():
    tick.set_rotation(30) 
plt.show()

# %%

# report 

def report(ret, indexset,f): ##this is for summarising investment results
    investresult_test = pd.DataFrame(index=[indexset],columns=['return','std','sharpe'])
    total_value = ret
    mean = total_value.mean()*f
    std = total_value.std()*np.sqrt(f)
    sharpe = mean/(std)

    investresult_test.loc[indexset,'return'] = mean
    investresult_test.loc[indexset,'std'] = std
    investresult_test.loc[indexset,'sharpe'] = sharpe
    return investresult_test

# %%
# Pring testing results of different methods
investresult = pd.concat([
    report(hed_res.stack(),'Hedge',52),
    report(eps_res.stack(),'EpsilonGreedy',52), 
    report(ucb_res.stack(),'UCB1',52), 
    report(mea_res,'MeanAverage',52), 
    report(pandl['totalreturn'],'Combined',52)], 
    ignore_index=False)

print(investresult)

# %%
