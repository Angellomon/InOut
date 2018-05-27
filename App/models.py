from django.db import models


class BannedItem(models.Model):
    ''' modelo del item que no esta disponible en el mercado '''
    nombre = models.CharField(max_length=50, blank=True, null=True)
    costo = models.FloatField(blank=True, null=True)
    ganancia = models.FloatField(blank=True, null=True)
    demanda = models.FloatField(blank=True, null=True)
    tiempo_mercado = models.FloatField(blank=True, null=True)
    valor_mercado = models.FloatField(blank=True, null=True)
    tiempo_fuera = models.FloatField(blank=True, null=True)
    valor_individual = models.FloatField(blank=True, null=True)
    def __str__(self):
        return '{}'.format(self.nombre)


class LiveItem(models.Model):
    ''' modelo del item que si esta disponible en el mercado '''
    nombre = models.CharField(max_length=50, blank=True, null=True)
    costo = models.FloatField(blank=True, null=True)
    ganancia = models.FloatField(blank=True, null=True)
    demanda = models.FloatField(blank=True, null=True)
    tiempo_mercado = models.FloatField(blank=True, null=True)
    valor_mercado = models.FloatField(blank=True, null=True)

    def __str__(self):
        return '{}'.format(self.nombre)
    