import numpy as np
import scipy.stats as sc

print('__________________ Task 1 __________________')
matrix = np.genfromtxt('matrix.txt', delimiter=',', dtype='int')
print(matrix)
print('Сумма всех элементов:', matrix.sum())
print('Максимальный элемент:', matrix.max())
print('Минимальный элемент:', matrix.min())
print('____________________________________________', '\n')

print('__________________ Task 2 __________________')
t2 = np.array([2, 2, 2, 4, 4, 3, 3, 3, 5, 6, 6, 7])
print(t2)
tmp1_t2, tmp2_t2 = [], []
counter = 1
for i in range(len(t2) - 1):
    if t2[i] == t2[i + 1]:
        counter += 1
        if i == len(t2) - 2:
            tmp1_t2.append(t2[i + 1])
            tmp2_t2.append(counter)
    else:
        tmp1_t2.append(t2[i])
        tmp2_t2.append(counter)
        counter = 1
        if i == len(t2) - 2:
            tmp1_t2.append(t2[i + 1])
            tmp2_t2.append(counter)
x1_t2, x2_t2 = np.array(tmp1_t2), np.array(tmp2_t2)
result_t2 = (x1_t2, x2_t2)
print(result_t2)
print('____________________________________________', '\n')

print('__________________ Task 3 __________________')
t3 = np.random.normal(10, 10, 40).reshape(10, 4)
print(t3)
print('Минимальное значение:', t3.min())
print('Максимальное значение:', t3.max())
print('Среднее значение:', t3.mean())
print('Стандартное отклонение:', t3.std())
substr5_t3 = t3[:5]
print(substr5_t3)
print('____________________________________________', '\n')

print('__________________ Task 4 __________________')
t4 = np.array([6, 2, 0, 3, 0, 0, 5, 7, 0])
print(t4)
tmp_t4 = []
for i in range(1, len(t4)):
    if t4[i - 1] == 0:
        tmp_t4.append(t4[i])
max_t4 = np.array(tmp_t4).max()
print('Максимальный элемент среди элементов, перед которыми стоит нулевой: ', max_t4)
print('____________________________________________', '\n')

print('__________________ Task 5 __________________')


def multivariate_normal_custom(X, m, C):
    N, D = X.shape
    delta = X - m
    invC = np.linalg.inv(C)
    log_detC = np.log(np.linalg.det(C))

    log_pdf = -0.5 * (D * np.log(2 * np.pi) + log_detC + np.einsum('ij,ij->i', delta, np.dot(invC, delta.T).T))
    return log_pdf


m = np.array([0, 0])  # мат. ожидание
C = np.array([[1, 0], [0, 1]])  # матрица ковариаций
X = np.array([[1, 2], [3, 4]])  # точки

result_custom = multivariate_normal_custom(X, m, C)
print("Результат реализованной функции:", result_custom)

result_scipy = sc.multivariate_normal(m, C).logpdf(X)
print("Результат функции scipy:", result_scipy)
print('____________________________________________', '\n')

print('__________________ Task 6 __________________')
t6 = np.arange(16).reshape(4, 4)
print(t6, '\n')
t6[[1, 3]] = t6[[3, 1]]
print(t6)
print('____________________________________________', '\n')

print('__________________ Task 7 __________________')
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
species = np.array([row.tolist()[4] for row in iris])
t7 = np.unique(species)
print('Уникальные значения:', t7)
print('Количество уникальных значений:', len(t7))
print('____________________________________________', '\n')

print('__________________ Task 8 __________________')
t8 = np.array([0, 1, 2, 0, 0, 4, 0, 6, 9])
print(t8)
print('Индексы ненулевых элементов:', np.flatnonzero(t8))
