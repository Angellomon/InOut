import numpy as np
import random

ERROR = .00001


def step(x):
    return 1 if x >= 0 else -1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def linear(x):
    return x


def generear_pesos(dim):
    # return [random.uniform(-1, 1) for i in range(dim)]
    return np.random.uniform(-1, 1, dim)


def normalizar(datos, lim_v, lim_n):
    return np.interp(datos, lim_v, lim_n)


def desnormalizar(datos, lim_v, lim_n):
    return np.interp(datos, lim_v, lim_n)


class Neuron():
    def __init__(self, name, outs=[], w=[], b=0):
        self.outs = np.array(outs)
        self.name = name
        self.w = np.array(w)
        self.b = b
        self.activation = sigmoid
        self.i = 0

    def __len__(self):
        return len(self.w)

    def activate_all(self, inputs):
        inputs = np.array(inputs)
        out = list()
        for i in inputs:
            # print(i)
            s = sum(i * self.w) + self.b
            out.append(self.activation(s))
            # print(out)
            # input()
        return out

    def activate(self, i):
        i = np.array(i)
        return self.activation(sum(i * self.w) + self.b)

    def __str__(self):
        return 'Neuron\n{}:\nw: {}, b: {}\noutputs: {}'.format(self.name, self.w, self.b, self.outs)


class Perceptron(Neuron):
    def __init__(self, name, outs=[], w=[], b=0):
        super(Perceptron, self).__init__(name, outs, w, b)
        self.activation = step

    def train(self, inputs):
        inputs = np.array(inputs)
        print(inputs)
        self.w = np.random.uniform(-100, 100, size=len(inputs[0]))
        self.b = np.random.uniform(-100, 100)
        # print(self.w, self.b)
        paro = False
        while not paro:
            paro = True
            i = 0
            act = self.activate_all(inputs)
            while i < len(inputs):
                if act[i] != self.outs[i]:
                    paro = False
                    j = 0
                    while j < len(inputs[0]):
                        self.w[j] += inputs[i][j] * self.outs[i]
                        j += 1
                    self.b += self.outs[i]
                    # break
                i += 1
        return self.w, self.b

    def __str__(self):
        return 'Perceptron\n{}:\nw: {}, b: {}\noutputs: {}'.format(self.name, self.w, self.b, self.outs)


class Adaline(Neuron):
    ''' clase que representa a una neurona tipo ADALINE '''

    def __init__(self, name, outs=[], w=[], b=0):
        ''' constructor de la neurona '''
        super(Adaline, self).__init__(name, outs, w, b)
        self.activation = linear
        self.RATE = .00001
        # self.err_ant = 10000000

    def __str__(self):
        ''' to_string '''
        return 'Adaline: {}\nw: {}, b: {}\noutputs:\n{}'.format(self.name, self.w, self.b, self.outs)

    def train(self, inputs):
        ''' implementación del algoritmo de entrenamiento basado en la 
            regla delta, que debe converger al error cuadratico medio '''
        inputs = np.array(inputs)
        self.w = np.random.uniform(-1, 1, size=len(inputs[0]))
        self.b = np.np.random.uniform(-1, 1)
        # print('inicial\nw: {}, b: {}'.format(self.w, self.b))
        while not self.paro(inputs):
            # self.difs.clear()
            i = 0
            # print('**************************')
            while i < len(inputs):
                # print('inputs: {}'.format(inputs[i]))
                d = self.outs[i]
                y = self.activate(inputs[i])
                # print(type(y))
                # input()
                # y = self.activate(inputs[i])
                dif = d - y

                # print(abs(dif))

                if abs(dif) > ERROR:
                    j = 0
                    while j < len(inputs[0]):
                        x_i = inputs[i][j]
                        self.w[j] += self.RATE * dif * x_i
                        j += 1
                    # print('w: {}'.format(self.w))
                    self.b += dif
                    # break
                i += 1
        return self.w, self.b

    def paro(self, inputs):
        c = 0
        r = 0
        i = 0
        p = 0
        # difs = list()
        while i < len(self.outs):
            dif = self.outs[i] - self.activate(inputs[i])
            # difs.append(dif)
            if abs(dif) > ERROR:
                c += 1
            p += abs(dif)
            dif = dif**2
            r += dif
            i += 1
        r = r / 2
        p = p / len(self.outs)
        # if self.err_ant < r:
        #     self.RATE *= 1.001
        # else:
        #     self.RATE /= 1.000001
        # self.err_ant = r
        # if self.i == 1000:
        # print('error: {}, difs: {}'.format(r, self.difs))
        c = round(c / len(self.outs), 4)
        print('w: {}, b: {},\t {},\t error: {}, {}%, prom: {}'.format(
            self.w, self.b, self.RATE, r, c * 100, p))
        # print(type(self.w[0]), type(self.b))
        # input()
        # input()
        # self.i = 0
        # input()
        if c == 0.0:
            return True
        else:
            return False


class PMonoNetwork:
    def __init__(self, name, outs=[], w=[], b=[]):
        self.D = np.array(outs)
        self.name = name
        self.W = np.array(w)
        self.B = np.array(b)
        self.RATE = 0.005
        self.f_activacion = linear

    def activar(self, inputs):
        inputs = np.array([inputs])
        # print(inputs)
        # print('****')
        # print(self.W)
        return np.matmul(inputs, self.W) + self.B

    def train(self, X):
        # print(X.T)
        X = np.array(X)
        # input(X)
        while not self.paro(X):
            for i, d in zip(X, self.D):
                Y = self.activar(i)
                DIF = d - Y
                # print(Y)
                if not np.all([abs(i) < ERROR for i in DIF]):
                    self.W += self.RATE * np.matmul(i.reshape(-1, 1), DIF)
                    self.B += DIF[0]
                # print('W: {}'.format(self.W))
                # print('B: {}'.format(self.B))
                # print('Y: {}\n'.format(Y))
            # print('DIF:\n{}\n'.format(DIF))
            # print('***********')
            # # input()
            # print('\n\n')

    def generear_pesos(self, inputs):
        inputs = np.array([inputs])
        self.W = generear_pesos((inputs.shape[1], self.D.shape[1]))
        self.B = np.random.uniform(-1, 1, self.D.shape[1])
        # print(self.W, self.B)

    def paro(self, inputs):
        c = 0
        for i, d in zip(inputs, self.D):
            Y = self.activar(i)
            DIF = d - Y
            gan = self.tomatodo(Y)
            if gan == np.argmax(d):
                c += 1

        # print('W: {}'.format(self.W))
        # print('Y: {}'.format(Y))
        # print('DIF: {}'.format(DIF))
        # R = [abs(i) < ERROR for i in DIF]
        # return np.all(R)
        return c == len(inputs)
    
    def tomatodo(self, Y):
        return np.argmax(Y)


class MultiCapa():
    def __init__(self, name, outs, capas=None, entradas=None, ws=[], bs=[]):
        # entradas += 1
        # self.WS = list()
        self.name = name
        self.D = np.array(outs)
        self.RATE = 1
        self.YS = list()
        self.DSIGS = list()
        # print(capas, 'que')
        # input(self.BS)
        if entradas and capas:
            self.BS = [np.random.randn(i, 1) for i in capas[0:]]
            self.WS = [np.random.randn(entradas, capas[0])]
            self.WS.extend([np.random.randn(i, j) for i, j in zip(capas[:-1], capas[1:])])
        else:
            self.WS = [np.array(i) for i in ws]
            self.BS = [np.array(i) for i in bs]
        # self.WS = np.array([[np.random.uniform(-1, 1), np.random.uniform(-1, 1)] for i in range(0, len(capas) + 1)])
        # self.WS = [
        #     np.array([[.1, .5], [-.7, .3]]),
        #     np.array([[.2], [.4]])
        # ]
        # print(self.WS)
        # input()
        # self.crear_pesos(capas)

    # def crear_pesos(self, capas, i):
    #     self.WS.append(generear_pesos((len(i[0]), capas[0])))
    #     i = 1
    #     while i < len(capas):
    #         self.WS.append(generear_pesos((capas[i - 1], capas[i])))
    #         i += 1

    def activar(self, inputs):
        self.YS.clear()
        self.DSIGS.clear()
        # print(self.BS)
        # inputs = np.array(inputs)
        # input(inputs)
        # O = sigmoid(inputs)
        # self.YS.append(O)
        # self.DSIGS.append(dsigmoid(O))
        self.YS.append(np.array(inputs))
        # O = sigmoid(np.matmul(inputs, self.WS[0]))
        # print('O: {}'.format(O))
        # self.YS.append(O)
        # self.DSIGS.append(dsigmoid(O))
        for i in range(0, len(self.WS)):
            # print('Y[i]: {}'.format(self.YS[i]))
            # print('WS[i]: {}'.format(self.WS[i]))
            # print('BS[i]: {}'.format(self.BS[i].T))
            # mul = 
            # r = np.zeros_like(mul)
            # for j in range(0, len(mul)):
            #     r[j]
            O = sigmoid(np.matmul(self.YS[i], self.WS[i]) + self.BS[i].T)
            # print(O)
            self.YS.append(O)
            # self.DSIGS.append(dsigmoid(O))
            i += 1
        # print('YS: {}'.format(self.YS))
        # input()
        return O

    def train(self, inputs):
        # print(self.WS)
        # input()
        # input(i)
        while not self.paro(inputs):
            DIFS = list()
            # YS = self.activar(inputs)
            
            # DIF = self.D - Y
            for i, d in zip(inputs, self.D):
                
                DIFS.clear()
                print('input: {}'.format(i))
                # feedforward
                # print('DSIGS: {}'.format(self.DSIGS))
                y = self.activar(i)
                print('YS: {}'.format(self.YS))
                print('y: {}'.format(y))
                print('d: {}'.format(d))
                print('WS: {}'.format(self.WS))
                print('BS: {}'.format(self.BS))
                if not self.check(i, d):
                    
                    DIF = d - y
                    DIFS.append(DIF)
                    j = 0
                    # obtencion error
                    for W in self.WS[::-1]:
                        # print('W: {}'.format(W))
                        # print('DIF: {}'.format(DIFS[j]))
                        DIFS.append(np.matmul(W, DIFS[j]))
                        j += 1
                    DIFS.pop()
                    # input()
                    DIFS = DIFS[::-1]
                    print('difs: {}'.format(DIFS))
                    # input(len(self.BS))
                    # fin obtencion error
                    # backpropagation
                    for j in range(0, len(self.WS)):
                    # while i < len(self.WS):
                        # print('*********************')
                        # print('j: {}'.format(j))
                        # print('WS[j]: {}'.format(self.WS[j]))
                        # print('DIFS[j]: {}'.format(DIFS[j]))
                        # print('DSIGS[j]: {}'.format(dsigmoid(self.YS[j + 1]).reshape((-1, 1)).T))
                        # print('YS[j]:\n{}'.format(self.YS[j].reshape((-1, 1))))
                        mul = np.matmul(self.YS[j].reshape((-1, 1)), dsigmoid(self.YS[j + 1]).reshape((-1, 1)).T)
                        mult = np.multiply(DIFS[j], mul)
                        # print('mul: {}'.format(mul))
                        # print('mult: {}'.format(mult))
                        # print('rate: {}'.format(self.RATE))
                        
                        self.WS[j] += np.multiply(self.RATE, mult)
                        # print('BS[j]: {}'.format(self.BS[j]))
                        # print(self.RATE * DIFS[j] * dsigmoid(self.YS[j + 1]))
                        self.BS[j] += DIFS[j].reshape((-1, 1)) * dsigmoid(self.YS[j + 1].reshape((-1, 1)))
                        
                    print('**********************')
                    input()

            # for i in range(0, len(YS[0])):
            # # for y, d in zip(YS, self.D):
            #     print('WS: {}'.format(self.WS))
            #     print('YS: {}'.format(YS))
            #     DIF = self.D[i] - YS[i]
            #     print(DIF)
            #     if abs(DIF) > ERROR:
            #         DIFS.append(DIF)
            #         j = 1
            #         # feedforward
            #         for W in self.WS[::-1]:
            #             print('W: {}'.format(W))
            #             print('DIF: {}'.format(DIFS[j - 1]))
            #             DIFS.append(np.matmul(W, DIFS[j - 1]))
            #             j += 1
            #         DIFS.pop()
            #         print('difs: {}'.format(DIFS))
            #         input()
            #         DIFS = DIFS[::-1]
            #         # backpropagation
            #         for j in range(0, len(self.WS)):
            #         # while i < len(self.WS):
            #             print('*********************')
            #             print('inputs: {}'.format(inputs))
            #             print('WS[i]: {}'.format(self.WS[j]))
            #             print('DIFS[i]: {}'.format(DIFS[j]))
            #             print('DSIGS[j][i]: {}'.format(self.DSIGS[j][i]))
            #             print('YS[j][i]:\n{}'.format(self.YS[j][i].reshape((-1, 1)).T))
            #             mul = np.matmul(self.YS[j][i].reshape(-1, 1), self.DSIGS[j][i].reshape(-1, 1).T)
            #             mult = np.multiply(np.array([DIFS[j]]).T, mul)
            #             print('mul: {}'.format(mul))
            #             print('mult: {}'.format(mult))
            #             print('rate: {}'.format(self.RATE))
                        
            #             self.WS[j] += np.multiply(self.RATE, mult)
            #             # print('WS[i]: {}'.format(self.WS[i]))
            #             print('**********************')
            #             input()
            # print('sale')



    def paro(self, inputs):
        Y = [self.check(i, d) for i, d in zip(inputs, self.D)]
        # Y = self.activar(inputs)
        # DIF = self.D - Y
        # print('W: {}'.format(self.W))
        # print('Y: {}'.format(Y))
        # print('DIF: {}'.format(DIF))
        # R = [abs(i) < ERROR for i in Y]
        return np.all(Y)
    
    def check(self, i, d):
        Y = self.activar(i)
        DIF = d - Y
        # print('W: {}'.format(self.W))
        # print('Y: {}'.format(Y))
        # print('DIF: {}'.format(DIF))
        R = [abs(j) < ERROR for j in DIF]
        return np.all(R)


class RedMulticapa():
    def __init__(self, tams):
        ''' constructor de la clase multicapa, recibe una tupla de tamaños
            que representan las capas en sí. '''
        self.num_capas = len(tams)
        self.tams = tams
        self.bs = [np.random.randn(i, 1) for i in tams[1:]]
        self.ws = [np.random.randn(i, j) for i, j in zip(tams[:-1], tams[1:])]
        input(self.ws)
        input(self.bs)
    
    def feedforward(self, i):
        ''' activacion feedforward con input "i" '''
        print('*************')
        for b, w in zip(self.bs, self.ws):
            i = sigmoid(np.matmul(i, w) + b)
            print('i: {}, w: {}'.format(i, w))
        input(i)
        return i

    def backpropagation(self, x, y):
        '''  '''
        n_bs = [np.zeros(b.shape) for b in self.bs]
        n_ws = [np.zeros(w.shape) for w in self.ws]
        # feedforward
        activacion = np.array(x)
        activaciones = [np.array(x)] # para guardar las activaciones
        zs = [] # para guardar los vectores z
        for b, w in zip(self.bs, self.ws):
            # print('w: {}'.format(w))
            # print('act: {}'.format(activacion))
            z = np.dot(w.T, activacion) + b
            zs.append(z)
            activacion = sigmoid(z)
            activaciones.append(activacion)
        delta = self.calcular_difs(activaciones[-1], y) * dsigmoid(zs[-1])
        n_bs[-1] = delta
        # print(activaciones)
        n_ws[-1] = np.dot(delta, activaciones[-2].transpose())

        for l in range(2, self.num_capas):
            z = zs[-l]
            dsig = dsigmoid(z)
            delta = np.dot(self.ws[-l + 1], delta) * dsig
            n_bs[-l] = delta
            n_ws[-l] = np.dot(delta, activaciones[-l - 1].transpose())
        return (n_bs, n_ws)
        
    
    def calcular_difs(self, activaciones, y):
        # print(activaciones - y)
        return (activaciones - y)

    def gradiente(self, datos, epocas, tam_lote, rate, test=None):
        ''' entrenamiento de la red multicapa a través del gradiente
            descendiente. '''
        if test:
            n_test = len(test)
        n = len(datos)
        # recorrido de epocas
        for j in range(epocas):
            random.shuffle(datos)
            mini_lotes = [
                datos[k:k+tam_lote] for k in range(0, n, tam_lote)
            ]
            for lote in mini_lotes:
                self.actualizar_lotes(lote, rate)
            if test:
                print('Epoca: {}: {} / {}'.format(j, self.evaluar(test), n_test))
            else:
                print('Epoca {} completada'.format(j))
    
    def actualizar_lotes(self, lote, rate):
        n_bs = [np.zeros(b.shape) for b in self.bs]
        n_ws = [np.zeros(w.shape) for w in self.ws]

        for x, y in lote:
            delta_b, delta_w = self.backpropagation(x, y)
            n_bs = [nb+dnb for nb, dnb in zip(n_bs, delta_b)]
            n_ws = [nw+dnw for nw, dnw in zip(n_ws, delta_w)]
        self.ws = [w-(rate/len(lote))*nw for w, nw in zip(self.ws, n_ws)]
        self.bs = [b-(rate/len(lote))*nb for b, nb in zip(self.bs, n_bs)]

    def evaluar(self, datos):
        ''' siempre debe cambiar, evalua a la red '''
        # input(datos)
        resultados = [
            (np.argmax(self.feedforward(x)), y) for (x, y) in datos
        ]
        return sum(int(x == y) for (x, y) in resultados)