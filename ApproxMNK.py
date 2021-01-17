import numpy as np
import matplotlib.pyplot as plt
import os

SAVE_PATH = os.getcwd()


class ApproxMNK:
    """ Для линейной ф-ции вида f(x) = kx + b """

    def __init__(self, n, sigma, k, b):
        self.N = n  # число экспериментов
        self.sigma = sigma  # стандартное отклонение наблюдаемых значений
        self.k = k  # теоретическое значение параметра k
        self.b = b  # теоретическое значение параметра b

    def get_func(self):
        """ Вычисление теоретической функции, моделирование наблюдений за счет случайных отклонений"""
        #   Дополнительный транспонированный вектор х
        x = np.array(range(self.N))
        #   Значения функции + отклонения
        f = np.array([self.k * z + self.b for z in range(self.N)])
        y = f + np.random.normal(0, self.sigma, self.N)
        return x, y, f

    def get_koef(self):
        """  Вычисление коэффициентов k и b по эксперементальным данным """
        x, y, f = self.get_func()
        mx = x.sum() / self.N
        my = y.sum() / self.N
        a2 = np.dot(x.T, x) / self.N
        a11 = np.dot(x.T, y) / self.N
        kk = (a11 - mx * my) / (a2 - mx ** 2)
        bb = my - kk * mx
        #   Точки аппроксимации
        ff = np.array([kk * z + bb for z in range(self.N)])
        return ff

    def get_err(self):
        """ Рассчет средней ошибки аппроксимации """
        x, y, f = self.get_func()
        ff = self.get_koef()
        err = round(sum([abs(f[i] - ff[i]) for i in range(0, len(x))]) / (len(x) * sum(f)) * 100, 4)
        return err

    def plot_graf(self):
        """ Построение графика аппроксимации """
        ff = self.get_koef()
        x, y, f = self.get_func()
        err = self.get_err()
        #   Средняя ошибка
        plt.scatter(x, y, s=2, c='green')
        plt.title('График аппроксимации линейной функции с помощью МНК')
        plt.xlabel('Средняя ошибка: {}%'.format(round(err * 100, 2)))
        plt.grid(True)
        plt.plot(f)
        plt.plot(ff, c='red')
        plt.legend(['Теоретическая функция', 'Аппроксимируемая функция', 'Точки аппроксимации'],
                   loc='upper center', bbox_to_anchor=(0.5, -0.13), shadow=True, ncol=2)
        plt.savefig(SAVE_PATH + '\grafik.png')
        plt.show()


if __name__ == '__main__':
    print('Введите N (целое) - число экспериментов , \n\t\tsigma - стандартное отклонение наблюдаемых значений, '
          '\n\t\tk - теоретическое значение параметра k, \n\t\tb - теоретическое значение параметра b')
    t = ApproxMNK(int(input()), float(input()), float(input()), float(input()))
    t.plot_graf()
