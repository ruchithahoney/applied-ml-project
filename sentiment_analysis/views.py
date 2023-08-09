from django.shortcuts import render
from django.http import JsonResponse
import json
import numpy as np
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras
from django.views.decorators.csrf import csrf_exempt

model = keras.models.load_model("sentiment_analysis.keras")

def get_sequences(text):
    #creating a tokenizer object
    tokenizer=Tokenizer()
    #using fit_on_text method to convert word into number most frequent would assign to 1 and with lower frequency assign to lower number
    tokenizer.fit_on_texts(text)
    #getting the word and the number assigning to them
    sequences=tokenizer.texts_to_sequences(text)

    #getting maximum length of list inthe sequences list

    print('Maximum Vocab',len(tokenizer.word_index))
    max_sequence_length=np.max(list(map(lambda x:len(x),sequences)))

    print('Max Sequences Length',max_sequence_length)

    sequences=pad_sequences(sequences,maxlen=max_sequence_length,padding='post')

    return sequences

def predict_value(text):
    output = model.predict(get_sequences(text))
    y_pred=np.argmax(output,axis=1)
    return y_pred

def index(request):
    return render(request, 'index.html')

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        predicted_value = list(predict_value(data.get("text")))
        pred_value = max(set(predicted_value), key = predicted_value.count)
        if pred_value == 1:
            answer = "POSITIVE"
        elif pred_value == 2:
            answer = "NEUTRAL"
        else:
            answer = "NEGATIVE"
        return JsonResponse({'predicted_class': answer})
