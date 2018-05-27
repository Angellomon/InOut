from django.urls import path
from App import views


app_name = 'app'

urlpatterns = [
    path('', views.index, name='home'),
    path('show-live/', views.mostrar_live, name='show-live'),
    path('show-banned/', views.mostrar_ban, name='show-ban'),
    path('consulta/', views.consulta, name='consulta'),
]