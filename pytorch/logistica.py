import numpy
import matplotlib.pyplot as pl


class Logistica:
    # regresion lineal
    def __init__(self):
        self.X = None
        self.y = None
        self.theta = None
        self.__historial = None

    # cargar variables características y variable objetivo
    def fit(self, x, y):
        m, n = x.shape
        columna_unos = numpy.ones((m, 1))
        self.X = numpy.append(columna_unos, x.reshape(m, -1), axis=1)
        self.y = y.reshape(-1, 1)
        self.theta = numpy.zeros(n + 1)

    def inicializar_parametros(self):
        m, n = self.X.shape
        self.theta = numpy.zeros(n)

    def normalizar(self):
        miu = self.X[:, 1:].mean(0)
        desviacion = self.X[:, 1:].std(0)
        self.X[:, 1:] = (self.X[:, 1:] - miu) / desviacion

    def sigmoide(self, z):
        g = 1 / (1 + numpy.exp(-z))
        return g

    # funcion costo (loss function)
    def get_j(self, theta):
        theta = theta.reshape(-1, 1)
        m = self.X.shape[0]
        h = self.sigmoide(self.X.dot(theta))
        j = -1 / m * (self.y.T.dot(numpy.log(h)) + (1 - self.y).T.dot(numpy.log(1 - h)))
        return j.sum()

    def getParametros(self):
        return self.theta

    # gradiente
    def get_gradiente(self, theta):
        theta = theta.reshape(-1, 1)
        m = self.X.shape[0]
        h = self.sigmoide(self.X.dot(theta))
        error = h - self.y
        t = 1 / m * self.X.T.dot(error)
        return t.flatten()

    def descenso_gradiente(self, alpha, epsilon=10e-6, itera=None):
        js = []
        theta = self.theta
        i = 0
        while True:
            js.append(self.get_j(theta))
            theta = theta - alpha * self.get_gradiente(theta)
            if abs(self.get_j(theta) - js[-1]) < epsilon:
                break
            elif itera is not None:
                if i >= itera:
                    break
            i = i + 1
        self.theta = theta
        self.__historial = numpy.array(js)

    def ecuacion_normal(self):
        self.theta = (
            numpy.linalg.pinv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)
        ).ravel()

    def graficar_historial(self):
        pl.plot(range(self.__historial.size), self.__historial)
        pl.grid()
        pl.xlabel("iteraciones")
        pl.ylabel(r"$J(\theta)$")
        pl.title("Evolución de costo en el descenso de Gradente")
        pl.show()

    # solo valido si solo se toma en cuenta una variable caracteristica
    def graficar_data(self, modelo=False):
        pos_i = numpy.where(self.y == 1)
        neg_i = numpy.where(self.y == 0)

        pl.scatter(self.X[pos_i, 1], self.X[pos_i, 2], c="red")
        pl.scatter(self.X[neg_i, 1], self.X[neg_i, 2], c="blue")
        if modelo:
            pl.plot(
                [0, -self.theta[0] / self.theta[2]],
                [-self.theta[0] / self.theta[1], 0],
                c="green",
            )
        pl.grid()
        pl.show()
