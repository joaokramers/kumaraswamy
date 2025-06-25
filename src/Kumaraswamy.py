import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

# Função densidade de probabilidade (PDF)
def kumaraswamy_pdf(x, a, b):
    return a * b * x**(a - 1) * (1 - x**a)**(b - 1)

# Função de distribuição acumulada (CDF)
def kumaraswamy_cdf(x, a, b):
    return 1 - (1 - x**a)**b

# Geração de amostras pela inversa da CDF
def kumaraswamy_rvs(a, b, size=1000):
    u = uniform.rvs(size=size)
    return (1 - (1 - u)**(1 / b))**(1 / a)

# Parâmetros da distribuição
a = 2.0
b = 5.0

# Espaço de valores de x para gráficos
x = np.linspace(0, 1, 1000)

# PDF e CDF
pdf = kumaraswamy_pdf(x, a, b)
cdf = kumaraswamy_cdf(x, a, b)

# Plot da PDF
plt.figure()
plt.plot(x, pdf, label='PDF')
plt.title(f'Distribuição de Kumaraswamy (a={a}, b={b})')
plt.xlabel('x')
plt.ylabel('Densidade')
plt.legend()
plt.grid(True)
plt.show()

# Plot da CDF
plt.figure()
plt.plot(x, cdf, label='CDF', color='orange')
plt.title(f'Função de distribuição acumulada (a={a}, b={b})')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.legend()
plt.grid(True)
plt.show()

# Geração de amostras e histograma
samples = kumaraswamy_rvs(a, b, size=10000)
plt.figure()
plt.hist(samples, bins=50, density=True, alpha=0.6, label='Amostras')
plt.plot(x, pdf, label='PDF teórica', linewidth=2)
plt.title('Amostragem da distribuição de Kumaraswamy')
plt.xlabel('x')
plt.ylabel('Frequência relativa')
plt.legend()
plt.grid(True)
plt.show()
