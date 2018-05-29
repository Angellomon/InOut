import tensorflow as tf
import json
import time

datos = json.load(open('datos/LiveCards.json', 'r'))

X = datos['inputs']
Y = datos['outs']

VENTANA = len(X[0])

x_ = tf.placeholder(tf.float32, shape=[len(X), VENTANA], name='X')
y_ = tf.placeholder(tf.float32, shape=[len(Y), 1], name='Y')

capa_0 = tf.Variable(tf.random_uniform([VENTANA, 20], -1, 1), name='capa0')
capa_1 = tf.Variable(tf.random_uniform([20, 10], -1, 1), name='capa1')
capa_2 = tf.Variable(tf.random_uniform([10, 1], -1, 1), name='capa2')

bias_0 = tf.Variable(tf.zeros([20]), name='bias0')
bias_1 = tf.Variable(tf.zeros([10]), name='bias1')
bias_2 = tf.Variable(tf.zeros([1]), name='bias2')

operacion = tf.sigmoid(tf.matmul(x_, capa_0) + bias_0)
operacion1 = tf.sigmoid(tf.matmul(operacion, capa_1) + bias_1)
hipotesis = tf.sigmoid(tf.matmul(operacion1, capa_2) + bias_2)

costo = tf.reduce_mean(((y_ * tf.log(hipotesis)) + ((1 - y_) * tf.log(1.0 - hipotesis))) * -1)

entrenamiento = tf.train.GradientDescentOptimizer(0.1).minimize(costo)

init = tf.global_variables_initializer()

sesion = tf.Session()
sesion.run(init)

x_graf = [i for i in range(250000)]
y_graf = list()

for i in range(250000):
    sesion.run(entrenamiento, feed_dict={
        x_: X,
        y_: Y
    })

    if i % 1000 == 0:
        print('Hipotesis: {}'.format(sesion.run(hipotesis, feed_dict={
            x_: X,
            y_: Y
        })))

        # print('Capa 0: {}'.format(sesion.run(capa_0)))
        # print('Bias 0: {}'.format(sesion.run(bias_0)))
        # print('Capa 1: {}'.format(sesion.run(capa_1)))
        # print('Bias 1: {}'.format(sesion.run(bias_1)))
        c = sesion.run(costo, feed_dict={
            x_: X,
            y_: Y
        })

        y_graf.append(float(c))

        print('Costo: {}'.format(c))
        print('Epoca: {}'.format(i))
ws = list()
ws.append(sesion.run(capa_0).tolist())
ws.append(sesion.run(capa_1).tolist())
ws.append(sesion.run(capa_2).tolist())
bs = list()
bs.append(sesion.run(bias_0).tolist())
bs.append(sesion.run(bias_1).tolist())
bs.append(sesion.run(bias_2).tolist())
# print(ws)
# print(bs)

json.dump({
    'ws': ws,
    'bs': bs,
    'x': x_graf,
    'y': list(y_graf)
}, open('datos/LiveCardsPesos.json', 'w'))