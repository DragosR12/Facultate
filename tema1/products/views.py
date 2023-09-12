from django.http import HttpResponse
from django.shortcuts import render

# /products -> inndex
# Uniform Resource Locator (Address)
def index(request):
    return HttpResponse('Hello World')

def new(request):
    return HttpResponse('New Products')
# Create your views here.
