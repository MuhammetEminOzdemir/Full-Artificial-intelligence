"""
02.12.2022
Asil Can Yılmaz
Erzurum Atatürk Üniversitesi 
3. Ders Kopya Kağıdı
"""
# %% Imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
a = 1 

x = np.arange(11)-5
y = a * x**2 + x

plt.plot(x,y,".-")
plt.grid(True)
plt.xlabel("Input x")
plt.ylabel("Output y")
plt.show()

def mse (aa, bb):
    return np.mean((aa-bb)**2)


l = []
for a in np.arange(11)-5:
    y_ = a * x**2 + x
    plt.plot(x,y_,".-", label= f"a = {a}")
    l.append(mse(y,y_))
    
plt.legend()
plt.show()

plt.plot(np.arange(11)-5 , l , ".-")
plt.xlabel("Parametre a")
plt.ylabel("MSE Fonksiyonuna Göre Hata")
plt.grid(True)

plt.show()


# %%

SINYAL_UZUNLUGU = 128

t = np.linspace(0, 2 * np.pi, SINYAL_UZUNLUGU)

sinyal = np.sin(t) + np.random.random((SINYAL_UZUNLUGU,)) * 0.1

plt.plot (sinyal)

# %%

gurultu = np.random.random((SINYAL_UZUNLUGU,))

plt.plot(gurultu)

# %%

plt.plot(sinyal)
plt.plot(gurultu)

# %%

import torch

# %%

hedef_tensor = torch.from_numpy(sinyal)
baslangic_tensor = torch.from_numpy(gurultu)

# %%

hedef_variable = torch.autograd.Variable(hedef_tensor, requires_grad=False)
baslangic_variable = torch.autograd.Variable(baslangic_tensor, requires_grad=True)

# %%

optimizer = torch.optim.Adam([baslangic_variable], lr=0.1)

# %%

loss_fn  = torch.nn.MSELoss()

# %%

update_no = 0
while update_no < 100:
    optimizer.zero_grad()

    loss = loss_fn(hedef_variable, baslangic_variable)
    
    loss.backward()
    
    optimizer.step()

    update_no += 1
    plt.clf()
    plt.plot(hedef_variable.detach().numpy())
    plt.plot(baslangic_variable.detach().numpy())
    plt.title(f"Update:{update_no},\n Loss Score: {loss}")
    plt.show(block=False)
    plt.pause(0.1)

# %%

# lr = 0.1
# lr = 0.01
# lr = 1

# %%

np.random.seed(0)

x = np.random.randint(-10,10,size=(10,))

y = -5 * x**2 + x * 3 + 25 # + np.random.randint(-100,100,size=(10,))

plt.scatter(x,y)
plt.grid(True)

# %%

x_variable = torch.autograd.Variable(torch.from_numpy(x).float(), requires_grad=False)
y_variable = torch.autograd.Variable(torch.from_numpy(y).float(), requires_grad=False)

a = torch.rand((1,))
b = torch.rand((1,))
c = torch.rand((1,))

a_variable = torch.autograd.Variable(a, requires_grad=True)
b_variable = torch.autograd.Variable(b, requires_grad=True)
c_variable = torch.autograd.Variable(c, requires_grad=True)

# %%

optimizer = torch.optim.Adam([a_variable,b_variable,c_variable], lr=1)

# %%

loss_fn  = torch.nn.MSELoss()

# %%
update_no = 0
x_ = np.linspace(-10,10,1000)

a_hepsi = []
b_hepsi = []
c_hepsi = []

while update_no < 100:
    
    optimizer.zero_grad()
    
    y_hat_variable = a_variable*x_variable**2 + b_variable*x_variable + c_variable

    loss = loss_fn(y_variable, y_hat_variable)
    
    loss.backward()
    
    optimizer.step()

    update_no += 1
    
    plt.clf()
    plt.scatter(x,y)

    y_ = a_variable.detach().numpy()*x_**2 + b.detach().numpy()*x_ + c_variable.detach().numpy()
    plt.plot(x_,y_)
    plt.title(f"Update:{update_no},\n Loss Score: {loss}")
    plt.show(block=False)
    plt.pause(0.1)
    a_hepsi.append(a_variable.detach().numpy().copy())
    b_hepsi.append(b_variable.detach().numpy().copy())
    c_hepsi.append(c_variable.detach().numpy().copy())
    
    
# %%

print(a_variable, b_variable, c_variable)

# %%

plt.plot(a_hepsi)
plt.show()

# %%

plt.plot(b_hepsi)
plt.show()

# %%

plt.plot(c_hepsi)
plt.show()

# %%

# %%

# %%
