import numpy as np
import matplotlib.pyplot as plt

yPareto = np.loadtxt("yPareto.txt")
initial_pareto = np.loadtxt("initial_pareto.txt")

New_Weights_ = np.loadtxt("New_Weights_.txt")
New_Paretos_ = np.loadtxt("New_Paretos_.txt")
New_Paretos_ = New_Paretos_[:-1]
print(yPareto)
print(New_Paretos_)


fig = plt.figure()
plt.plot(yPareto[:,0],yPareto[:,1],"or",markersize = 8,label='solutions we found')
plt.plot(initial_pareto[:,0],initial_pareto[:,1],"*b",markersize = 15,label='initial observation')
plt.legend('Model length', 'Data length',loc='upper center', shadow=True)
plt.xlabel('f1')
plt.ylabel('f2')
plt.legend()
plt.title('r2>r1')
#plt.savefig('2_32_lONG_times.pdf', format='pdf', dpi=1000)
plt.show()



fig = plt.figure()
for i in range(len(New_Paretos_)):
    plt.plot(New_Paretos_[i,0],New_Paretos_[i,1],'ob',marker='*',markersize=12)
    plt.text(New_Paretos_[i,0]-0.5,New_Paretos_[i,1]-0.5,str(np.round(New_Weights_[i],10)),size=14)

plt.xlabel('f1')
plt.ylabel('f2')
plt.title('Check out the weights')
plt.show()


