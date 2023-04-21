from django.http import JsonResponse, HttpResponseBadRequest
import requests
from django.http import HttpResponse
import numpy as np
import cv2
from skimage.metrics import structural_similarity


#Per testare lo stato in modo semplice la connessione all'api.
#Se restituisce 'Ok' allora c'è connessione e il server  online
def connection_api(request):
    if request.method == 'GET':
        try:
            return JsonResponse({'Status': 'Ok'})
        except requests.exceptions.RequestException:
            return HttpResponse('Error during API request')
        except ValueError:
            return HttpResponse('Invalid response format')
    else:
        return HttpResponse('Invalid request method')





# Metodo per la richiesta process_image con body due immagini.
# Controlla che ci siano le due immagini e chiama il metodo comparison_face
# Restituisce lo score alla chiamata http
def process_image(request):
    if request.method == 'POST':
        # Verifica se le immagini sono presenti nella richiesta
        if 'photoDB' not in request.FILES or 'photoNow' not in request.FILES:
            return HttpResponseBadRequest('Problems with your photos')

        # Leggi i dati binari delle immagini
        photo_db = request.FILES['photoDB'].read()
        photo_now = request.FILES['photoNow'].read()

        # Controlla che le immagini siano state correttamente ricevute
        if not photo_db or not photo_now:
            return HttpResponseBadRequest('Problems with your photos')

        # Decodifica l'immagine utilizzando cv2.imdecode
        nparr_db = np.frombuffer(photo_db, np.uint8)
        nparr_now = np.frombuffer(photo_now, np.uint8)
        img_db = cv2.imdecode(nparr_db, cv2.IMREAD_COLOR)
        img_now = cv2.imdecode(nparr_now, cv2.IMREAD_COLOR)

        # Ritaglia le immagini del volto
        photo_db = crop_face(img_db)
        photo_now = crop_face(img_now)

        # Controlla le dimensioni delle immagini
        #if photo_db.shape != photo_now.shape:
        #    return HttpResponseBadRequest('Images have different dimensions')

        print("Images are good")
        # Confronto tra le due photo con OpenCV
        ssim = comparison_face(photo_db, photo_now)
        print("Score: " + str(ssim))

        # Restituisci la risposta di quanto si assomigliano i visi
        return HttpResponse(str(ssim))

    # Se la richiesta non è una POST, restituisci un errore
    return HttpResponseBadRequest('Richiesta non valida')


def crop_face(image):
    # Usa il classificatore di Haar per rilevare il volto
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Verifica se è stato trovato un solo volto
    if len(faces) == 1:
        # Ritaglia l'immagine per avere solo il volto
        (x, y, w, h) = faces[0]
        cropped_image = image[y:y + h, x:x + w]

        # Adatta le immagini alla stessa dimensione
        cropped_image = cv2.resize(cropped_image, (256, 256))

        return cropped_image
    else:
        # Restituisci un errore se sono stati trovati più o nessun volto
        raise Exception('Unable to detect face or multiple faces detected')



def comparison_face(photo_db, photo_now):
    # Converte le immagini in scala di grigi
    gray_db = cv2.cvtColor(photo_db, cv2.COLOR_BGR2GRAY)
    gray_now = cv2.cvtColor(photo_now, cv2.COLOR_BGR2GRAY)

    # Calcola l'indice SSIM tra le due immagini
    ssim = structural_similarity(gray_db, gray_now)

    return ssim