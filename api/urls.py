from django.urls import path
from . import ekyc

urlpatterns = [
    path('', ekyc.index),
    path('process', ekyc.process),
]
