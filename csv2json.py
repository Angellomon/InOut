import csv
import json


nombre = 'LiveCardsPruebas'
datos = list(csv.reader(open('datos/{}.csv'.format(nombre), 'r')))
# cars = datos[0]
caracteristicas = dict()

caracteristicas['inputs'] = list()
caracteristicas['outs'] = list()

# for c in cars:
#     caracteristicas[c.strip()] = []

for row in datos:
    try:
        l = list()
        for el in row[:-1]:
            l.append(float(el))
        caracteristicas['inputs'].append(l)
        caracteristicas['outs'].append([int(row[-1])])
    except Exception:
        continue
json.dump(caracteristicas, open('datos/{}.json'.format(nombre), 'w'))
