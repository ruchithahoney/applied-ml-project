from django.shortcuts import render
from django.http import JsonResponse
import base64
import numpy as np
import cv2
from django.views.decorators.csrf import csrf_exempt


def index(request):
    return render(request, 'index.html')

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        return JsonResponse({'predicted_class': "true"})
