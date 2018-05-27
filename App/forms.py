from django import forms
from App.models import LiveItem, BannedItem


class FormLiveI(forms.ModelForm):
    class Meta():
        model = LiveItem
        fields = '__all__'

class FormBannedI(forms.ModelForm):
    class Meta():
        model = BannedItem
        fields = '__all__'

class SelectLiveI(forms.Form):
    items = forms.ModelChoiceField(LiveItem.objects.order_by('nombre'))

class SelectBannedI(forms.Form):
    items = forms.ModelChoiceField(BannedItem.objects.order_by('nombre'))