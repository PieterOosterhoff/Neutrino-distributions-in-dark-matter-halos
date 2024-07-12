import numpy as np
import scipy.special as sp
import scipy.integrate as intr
import matplotlib.pyplot as plt

global GeV_per_c2_per_cm3
global g_n
global h
global kpc
global Gev
global c

G = 6.674*10**-11
g_n =4
h = 6.626*10**-34
h_bar = h/(2*np.pi)
c=2.9979*10**8
kpc=3.0857*10**19
eV_per_c2_per_m3 = 1.602*10**(-19)*c**(-2)
GeV_per_c2_per_cm3 = 10**15*eV_per_c2_per_m3

M_sun = 1.989*10**30

rho_0 = 0.184*GeV_per_c2_per_cm3
r_ster = 2*kpc
r_c = 24.42*kpc

plt.clf()
def massaminimum(alpha, beta, gamma, rho_0, r_ster, r_c):
    Term1=(2*h**3)/g_n
    Term2=((8*np.pi**2*G*r_c**2*rho_0**(1/3))/(3-gamma))**(-3/2)
    Term3=((r_ster/r_c)**gamma*(1+(r_ster/r_c)**alpha)**((beta-gamma)/alpha))**(-5/2)
    Term4=(intr.quad(lambda x: sp.hyp2f1((3-gamma)/alpha,round(((beta-gamma)/alpha),12),round(((3-gamma+alpha)/alpha),12),-x**alpha)/((1+x**alpha)**((beta-gamma)/alpha)*x**(2*gamma-1)), (r_ster/r_c), np.inf)[0])**(-3/2) #Rounding to 12 decimals is necessary to make the integral behave properly

    m_min = (Term1*Term2*Term3*Term4)**(1/4)*(c**2/(1.602*10**-19))
    return m_min

def rho(r,r_s,rho_0,alpha,beta,gamma):
    x = r/r_s
    left = x**gamma
    right = (1+(x)**alpha)**((beta-gamma)/alpha)
    return rho_0/(left*right)

def frac(r,r_s,rho_0,alpha,beta,gamma):
    y=r/r_s
    rho_r = rho(r,r_s,rho_0,alpha,beta,gamma)
    Hyp2F1 = sp.hyp2f1((3-gamma)/alpha,round(((beta-gamma)/alpha),12),round(((3-gamma+alpha)/alpha),12),-(r/r_s)**alpha)
    m_n = 10*10**(-3)*(c**2/(1.602*10**-19))**(-1)

    Term1 = g_n*m_n**4/(2*h_bar**3)
    Term2 = y**gamma*(1+y**alpha)**((beta-gamma)/alpha)
    Term3 = 2*G*rho_0**(1/3)*r_s**2/(3-gamma)
    Term4 = intr.quad(lambda x: sp.hyp2f1((3-gamma)/alpha,round(((beta-gamma)/alpha),12),round(((3-gamma+alpha)/alpha),12),-x**alpha)/(x**(2*gamma-1)*(1+x**alpha)**((beta-gamma)/alpha)), r/r_s, np.inf)[0]
    #return Term3**(3/2)
    return Term1*Term2**(5/2)*(Term3*Term4)**(3/2)

def rho_nu(r,r_s,rho_0,alpha,beta,gamma):
    rho_r = rho(r,r_s,rho_0,alpha,beta,gamma)
    Hyp2F1 = sp.hyp2f1((3-gamma)/alpha,round(((beta-gamma)/alpha),12),round(((3-gamma+alpha)/alpha),12),-(r/r_s)**alpha)
    m_n = 10*10**(-3)*(c**2/(1.602*10**-19))**(-1)

    Term1 = g_n*m_n**4/(2*h_bar**3)
    Term2 = 2*G*rho_0**2*r_s**2/(rho_r*(3-gamma))
    Term3 = intr.quad(lambda x: sp.hyp2f1((3-gamma)/alpha,round(((beta-gamma)/alpha),12),round(((3-gamma+alpha)/alpha),12),-x**alpha)/(x**(2*gamma-1)*(1+x**alpha)**((beta-gamma)/alpha)), r/r_s, np.inf)[0]
    return Term1*(Term2*Term3)**(3/2)

def makeplot(Clusterno,max_r,points,alpha,beta,gamma):
    fraclist =[]
    frac2list=[]
    Flist = []
    rho_h_list = []
    rho_nu_list = []
    Calcnumber = points
    Maxradius = max_r*r_s[Clusterno]
    testradius = list(np.linspace(r_ster[Clusterno],Maxradius,Calcnumber))
    for i in range(Calcnumber):
        fraclist.append(frac(testradius[i],r_s[Clusterno],rho_0[Clusterno],alpha,beta,gamma))
        rho_h_list.append(rho(testradius[i],r_s[Clusterno],rho_0[Clusterno],alpha,beta,gamma)/rho_0[Clusterno])
        rho_nu_list.append(rho_nu(testradius[i],r_s[Clusterno],rho_0[Clusterno],alpha,beta,gamma)/rho_0[Clusterno])
        Flist.append(f_R(r_s[Clusterno],rho_0[Clusterno],alpha,beta,gamma,testradius[i]))
    plt.xlabel('$r$ (in kpc)',fontsize=15)
    testradius = np.array(testradius)
    testradius = testradius/kpc

    plt.yscale('log')
    plt.title(profiles[prof]+' profile', fontsize=15)
    plt.plot(testradius,fraclist,label=Clusters[Clusterno]+': $f(r)$')
    plt.plot(testradius,rho_h_list,label=Clusters[Clusterno]+': $rho_h(r)/\\rho_0$' )
    plt.plot(testradius,rho_nu_list,label=Clusters[Clusterno]+': $rho_\\nu(r)/\\rho_0$')
    plt.plot(testradius,Flist,color='brown',label=Clusters[Clusterno]+': $F(r)$')
    plt.axvline(x=r_s[Clusterno]/kpc,color='r',label=Clusters[Clusterno]+': $r_s$').set_dashes([2,2])
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.legend(prop={'size': 15})

def f_R(r_s,rho_0,alpha,beta,gamma,R):
    top = intr.quad(lambda x: np.pi*x**2*frac(x,r_s,rho_0,alpha,beta,gamma)*rho(x,r_s,rho_0,alpha,beta,gamma),0,R)[0]
    bottom = intr.quad(lambda y: np.pi*y**2*rho(y,r_s,rho_0,alpha,beta,gamma),0,R)[0]
    return top/bottom
'''
To generate a graph, insert the following at the relevant point. CL refers to which cluster you're graphing.
CL = 0
print(Clusters[CL]+': '+str(f_R3(r_s[CL],rho_0[CL],alpha[CL],beta[CL],gamma[CL],4*r_s[CL])))
makeplot(CL,10,1000,alpha[CL],beta[CL],gamma[CL])
plt.show()
'''

#Data from Meritt et al.
profiles = np.array(['$(1,3,\\gamma$)','Anisotropic Dehnen & McLaughlin','Isotropic Dehnen & McLaughlin','NFW'])
Clusters=np.array(['A09','B09','C09','D12','E09','F09'])
Galaxies = np.array(['G00','G01','G02','G03'])

#(1,3,gamma) model
print('(1,3,gamma) model (Table 1):')
prof = 0
#Clusters (1,3,gamma)

alpha = [1,1,1,1,1,1]
beta = [3,3,3,3,3,3]
gamma=np.array([1.174,1.304,0.896,1.251,1.265,1.012])

r_s=np.array([626.9,1164,241.8,356.1,382.5,233.9])
r_s = r_s*kpc

rho_s=np.array([3.87,4.75,3.27,3.82,3.96,3.51])
rho_s=10**(-rho_s)
for i in range(len(gamma)):
    rho_s[i]=2**((beta[i]-gamma[i])/alpha[i])*rho_s[i]
rho_0=rho_s
rho_0=rho_0*M_sun/((10**-3*kpc)**3)

r_ster=np.array([73,87,73,64,73,72])
r_ster=10**(r_ster/75)*kpc


for i in range(len(r_ster)):
    print(Clusters[i]+': '+str(round(massaminimum(alpha[i],beta[i],gamma[i],rho_0[i],r_ster[i],r_s[i]),3)))
print('')
#Galaxies (1,3,gamma)

alpha = [1,1,1,1]
beta = [3,3,3,3]
gamma=np.array([1.163,1.275,1.229,1.593])

r_s=np.array([27.96,35.34,53.82,54.11])
r_s = r_s*kpc

rho_s=np.array([3.16,3.36,3.59,3.70])
rho_s=10**(-rho_s)
for i in range(len(gamma)):
    rho_s[i]=2**((beta[i]-gamma[i])/alpha[i])*rho_s[i]
rho_0=rho_s
rho_0=rho_0*M_sun/((10**-3*kpc)**3)

r_ster=np.array([39,39,39,39])  #resolved radius
r_ster=10**(r_ster/128)*kpc

for i in range(len(r_ster)):
    print(Galaxies[i]+': '+str(round(massaminimum(alpha[i],beta[i],gamma[i],rho_0[i],r_ster[i],r_s[i]),3)))
print('')
print('')
#Anisotropic Dehnen & McLaughlin
print('Anisotropic Dehnen & McLaughlin (Table 2):')
prof = 1
#Clusters Anisotropic Dehnen & McLaughlin
gamma=np.array([0.694,0.880,0.241,0.683,0.669,0.350])
alpha = (3-gamma)/5
beta = (18-gamma)/5

r_s=np.array([722.7,1722,207,322.8,330.4,193.6])
r_s = r_s*kpc


rho_s=np.array([2.21,3.30,1.34,1.95,2.04,1.56])
rho_s=10**(-rho_s)
for i in range(len(gamma)):
    rho_s[i]=2**((beta[i]-gamma[i])/alpha[i])*rho_s[i]
rho_0=rho_s
rho_0=rho_0*M_sun/((10**-3*kpc)**3)

r_ster=np.array([76,92,76,67,77,76])
r_ster=10**(r_ster/80)*kpc
for i in range(len(r_ster)):
    print(Clusters[i]+': '+str(round(massaminimum(alpha[i],beta[i],gamma[i],rho_0[i],r_ster[i],r_s[i]),3)))

print('')

#Galaxies Anisotropic Dehnen & McLaughlin
gamma=np.array([0.422,0.568,0.581,0.849])
alpha = (3-gamma)/5
beta = (18-gamma)/5

r_s=np.array([20.89,25.88,43.05,30.20])
r_s = r_s*kpc

rho_s=np.array([1.11,1.28,1.60,1.34])
rho_s=10**(-rho_s)
for i in range(len(gamma)):
    rho_s[i]=2**((beta[i]-gamma[i])/alpha[i])*rho_s[i]
rho_0=rho_s
rho_0=rho_0*M_sun/((10**-3*kpc)**3)

r_ster=np.array([36,38,38,37])
r_ster=10**(r_ster/122)*kpc


for i in range(len(r_ster)):
    print(Galaxies[i]+': '+str(round(massaminimum(alpha[i],beta[i],gamma[i],rho_0[i],r_ster[i],r_s[i]),3)))
print('')



#Isotropic Dehnen & McLaughlin
print('Isotropic Dehnen & McLaughlin (Table 3):')
prof = 2
alpha = [4/9,4/9,4/9,4/9,4/9,4/9]
beta = [31/9,31/9,31/9,31/9,31/9,31/9]
gamma = [7/9,7/9,7/9,7/9,7/9,7/9]
#Clusters Isotropic Dehnen & McLaughlin
r_s=np.array([933.7,1180,554.3,409.1,428.2,438.2])
r_s = r_s*kpc


rho_s=np.array([2.43,2.97,2.27,2.17,2.28,2.32])
rho_s=10**(-rho_s)
for i in range(len(gamma)):
    rho_s[i]=2**((beta[i]-gamma[i])/alpha[i])*rho_s[i]
rho_0=rho_s
rho_0=rho_0*M_sun/((10**-3*kpc)**3)

r_ster=np.array([145,176,145,126,144,144])
r_ster=10**(r_ster/152)*kpc

for i in range(len(r_ster)):
    print(Clusters[i]+': '+str(round(massaminimum(alpha[i],beta[i],gamma[i],rho_0[i],r_ster[i],r_s[i]),3)))

print('')
#Galaxies Isotropic Dehnen & McLaughlin
r_s=np.array([34.43,36.53,63.06,26.98])
r_s = r_s*kpc


rho_s=np.array([1.59,1.61,1.97,1.23])
rho_s=10**(-rho_s)
for i in range(len(rho_s)):
    rho_s[i]=2**((beta[i]-gamma[i])/alpha[i])*rho_s[i]
rho_0=rho_s
rho_0=rho_0*M_sun/((10**-3*kpc)**3)

r_ster=np.array([46,46,46,47])
r_ster=10**(r_ster/151)*kpc

for i in range(len(r_ster)):
    print(Galaxies[i]+': '+str(round(massaminimum(alpha[i],beta[i],gamma[i],rho_0[i],r_ster[i],r_s[i]),3)))

print('')
#NFW
prof = 3
print('NFW (Table 3):')

alpha = [1,1,1,1,1,1]
beta = [3,3,3,3,3,3]
gamma = [1,1,1,1,1,1]
#NFW clusters
r_s=np.array([419.8,527.2,284.4,213.3,227.0,229.0])
r_s = r_s*kpc


rho_s=np.array([3.5,4.03,3.42,3.34,3.46,3.49])
rho_s=10**(-rho_s)
for i in range(len(gamma)):
    rho_s[i]=2**((beta[i]-gamma[i])/alpha[i])*rho_s[i]
rho_0=rho_s
rho_0=rho_0*M_sun/((10**-3*kpc)**3)

r_ster=np.array([161,195,161,139,161,160])
r_ster=10**(r_ster/168)*kpc

for i in range(len(r_ster)):
    print(Clusters[i]+': '+str(round(massaminimum(alpha[i],beta[i],gamma[i],rho_0[i],r_ster[i],r_s[i]),3)))

print('')
r_s=np.array([22.23,23.12,36.39,19.54])
r_s = r_s*kpc


rho_s=np.array([2.94,2.95,3.22,2.68])
rho_s=10**(-rho_s)
for i in range(len(r_s)):
    rho_s[i]=2**((beta[i]-gamma[i])/alpha[i])*rho_s[i]
rho_0=rho_s
rho_0=rho_0*M_sun/((10**-3*kpc)**3)

r_ster=np.array([52,50,51,51])
r_ster=10**(r_ster/167)*kpc

for i in range(len(r_ster)):
    print(Galaxies[i]+': '+str(round(massaminimum(alpha[i],beta[i],gamma[i],rho_0[i],r_ster[i],r_s[i]),3)))