import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'InOut.settings')

import django
django.setup()

import json

from App.models import BannedItem, LiveItem

def agregar_live(nombre, cars):
    item = LiveItem.objects.get_or_create(nombre=nombre, costo=cars[0], ganancia=cars[1], demanda=cars[2], tiempo_mercado=cars[2], valor_mercado=cars[4])[0]
    item.save()
    return item

def llenar_live(nombres, caracteristicas):
    for nombre, cars in zip(nombres, caracteristicas):
        agregar_live(nombre, cars)

def agregar_ban(nombre, cars):
    item = BannedItem.objects.get_or_create(nombre=nombre, costo=cars[0], ganancia=cars[1], demanda=cars[2], tiempo_mercado=cars[2], valor_mercado=cars[4], tiempo_fuera=cars[5], valor_individual=cars[6])[0]
    item.save()
    return item

def llenar_ban(nombres, caracteristicas):
    for nombre, cars in zip(nombres, caracteristicas):
        agregar_ban(nombre, cars)

def main():
    # live
    datos = json.load(open('static/datos/LiveCardsPruebas.json', 'r'))
    nombres = list()

    with open('static/datos/LiveCards.txt', 'r') as arch:
        for row in arch:
            nombres.append(str(row).strip().title())

    llenar_live(nombres, datos['inputs'])

    # ban
    datos = json.load(open('static/datos/BannedCardsPruebas.json', 'r'))
    nombres = list()

    with open('static/datos/BannedCards.txt', 'r') as arch:
        for row in arch:
            nombres.append(str(row).strip().title())
        
    llenar_ban(nombres, datos['inputs'])

if __name__ == '__main__':
    print('comenzando')
    main()
    print('terminado')