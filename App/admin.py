from django.contrib import admin
from App.models import LiveItem, BannedItem


admin.site.register(LiveItem)
admin.site.register(BannedItem)
