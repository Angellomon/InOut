from django.shortcuts import render, redirect
from App.models import LiveItem, BannedItem
from App.forms import SelectBannedI, SelectLiveI, FormBannedI, FormLiveI
import Neural.neural as nl
import json


def index(request):
    return render(request, 'App/index.html')

def mostrar_live(request):
    items_live = LiveItem.objects.order_by('nombre')
    cntxt = {
        'itemslive':items_live
    }
    return render(request, 'App/mostrar-live.html', context=cntxt)

def mostrar_ban(request):
    items_ban = BannedItem.objects.order_by('nombre')
    cntxt = {
        'itemsban': items_ban
    }
    return render(request, 'App/mostrar-ban.html', context=cntxt)

def consulta(request):
    sel_live = SelectLiveI()
    sel_banned = SelectBannedI()
    form_live = FormLiveI()
    form_banned = FormBannedI()

    if request.method == 'POST':
        if 'sel-li' in request.POST:
            form = SelectLiveI(request.POST)
            if form.is_valid():
                instance = LiveItem.objects.get(pk=form.cleaned_data['items'].pk)
                r = calcular_hipotesis_live(instance)
                return resultado(request, r)
        elif 'sel-ban' in request.POST:
            form = SelectBannedI(request.POST)
            if form.is_valid():
                instance = BannedItem.objects.get(pk=form.cleaned_data['items'].pk)
                r = calcular_hipotesis_ban(instance)
                return resultado(request, r)
        elif 'form-li' in request.POST:
            form = FormLiveI(request.POST)
            if form.is_valid():
                costo = form.cleaned_data['costo']
                ganancia = form.cleaned_data['ganancia']
                demanda = form.cleaned_data['demanda']
                tiempo_mercado = form.cleaned_data['tiempo_mercado']
                valor_mercado = form.cleaned_data['valor_mercado']

                item = LiveItem(costo=costo, ganancia=ganancia, demanda=demanda, tiempo_mercado=tiempo_mercado, valor_mercado=valor_mercado)

                r = calcular_hipotesis_live(item)
                return resultado(request, r)
        elif 'form-ban' in request.POST:
            form = FormBannedI(request.POST)
            if form.is_valid():
                costo = form.cleaned_data['costo']
                ganancia = form.cleaned_data['ganancia']
                demanda = form.cleaned_data['demanda']
                tiempo_mercado = form.cleaned_data['tiempo_mercado']
                valor_mercado = form.cleaned_data['valor_mercado']
                tiempo_fuera = form.cleaned_data['tiempo_fuera']
                valor_individual = form.cleaned_data['valor_individual']

                item = BannedItem(costo=costo, ganancia=ganancia, demanda=demanda, tiempo_mercado=tiempo_mercado, valor_mercado=valor_mercado, tiempo_fuera=tiempo_fuera, valor_individual=valor_individual)

                r = calcular_hipotesis_ban(item)
                return resultado(request, r)

    cntxt = {
        'sel_live': sel_live,
        'sel_banned': sel_banned,
        'form_live': form_live,
        'form_banned': form_banned
    }
    return render(request, 'App/consulta.html', context=cntxt)

def calcular_hipotesis_live(item):
    costo = item.costo
    ganancia = item.ganancia
    demanda = item.demanda
    tiempo_mercado = item.tiempo_mercado
    valor_mercado = item.valor_mercado
    inputs = [costo, ganancia, demanda, tiempo_mercado, valor_mercado]

    datos = json.load(open('static/datos/LiveCardsPesos.json', 'r'))
    ws = datos['ws']
    bs = datos['bs']

    net = nl.MultiCapa('live', [], ws=ws, bs=bs)

    y = round(net.activar(inputs)[0], 4) * 100

    return y

def calcular_hipotesis_ban(item):
    costo = item.costo
    ganancia = item.ganancia
    demanda = item.demanda
    tiempo_mercado = item.tiempo_mercado
    valor_mercado = item.valor_mercado
    tiempo_fuera = item.tiempo_fuera
    valor_individual = item.valor_individual
    inputs = [costo, ganancia, demanda, tiempo_mercado, valor_mercado, tiempo_fuera, valor_individual]

    datos = json.load(open('static/datos/BannedCardsPesos.json', 'r'))
    ws = datos['ws']
    bs = datos['bs']

    net = nl.MultiCapa('banned', [], ws=ws, bs=bs)

    y = round(net.activar(inputs)[0], 4) * 100

    return y

def resultado(request, y):
    return render(request, 'App/resultado.html', context={'resultado':y})