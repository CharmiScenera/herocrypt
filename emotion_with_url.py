import pandas as pd
import pymongo
import numpy as np
import ssl
import numpy as np
import cv2
from flask_cors import CORS
from PIL import Image
import requests
import io
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import FaceAttributeType
import json

def run_emotion_recognition(
    image_uri : str,
    face_db_uri : str,
    face_client_endpoint : str,
    face_client_key : str
    ):

    try:
        myclient = pymongo.MongoClient(
            face_db_uri,
            ssl=True,
            ssl_cert_reqs=ssl.CERT_NONE
            )
        mydb = myclient["New_face"]
        mycol = mydb["faceInfo"]
        users = mycol.find()
        users = pd.DataFrame(list(users))
        active_frame = users[users['Active']==True]
        active_frame = active_frame.reset_index(drop=True)

    except Exception as error_message:
        print("Error:", error_message)
        return {'ProcessingStatus': "Error", "Error_Message": error_message}

    try:
        img = Image.open(requests.get(image_uri, stream=True).raw)
        img = np.array(img)
        _, buf = cv2.imencode('.png', img)
        stream = io.BytesIO(buf)

        features = [
            FaceAttributeType.age,
            FaceAttributeType.emotion,
            FaceAttributeType.glasses]

        print("\nFeatures: ", features)

    except Exception as error_message:
        print("Error:", error_message)
        return {'ProcessingStatus': "Error", 'Error_Message': error_message}

    try:
        face_client = FaceClient(
            face_client_endpoint,
            CognitiveServicesCredentials(face_client_key))
        detected_faces = face_client.face.detect_with_stream(
            stream,return_face_id=True,
            return_face_attributes=features
            )

        print("\nDetected faces: ", detected_faces)

    except Exception as error_message:
        print("Error:", error_message)
        return {'ProcessingStatus': "Error", "Error_Message": error_message}

    if detected_faces:
        for face in detected_faces:
            #print('\nFace ID: {}'.format(face.face_id))
            detected_attributes = face.face_attributes.as_dict()

            #print_emotion_results(detected_attributes)

            detected_attributes['face_id'] = face.face_id

            if detected_attributes:
                return {'ProcessingStatus': "Detected", 'Values': detected_attributes}
            else:
                return {'ProcessingStatus': "Undetected", 'Error_Message': "Image does not contain any emotion"}
    else:
        print("\n No faces detected!\n")
        return {'ProcessingStatus': "Undetected", 'Error_Message': "No faces detected at all."}

def print_emotion_results(detected_attributes):
    age = 'age unknown' if 'age' not in detected_attributes.keys() else int(detected_attributes['age'])
    print(' - Age: {}'.format(age))
    if 'emotion' in detected_attributes:
        print(' - Emotions:')
        for emotion_name in detected_attributes['emotion']:
            print('   - {}: {}'.format(emotion_name, detected_attributes['emotion'][emotion_name]))
    if 'glasses' in detected_attributes:
        print(' - Glasses:{}'.format(detected_attributes['glasses']))

#img_url = ['https://dpdevscenerasd1.blob.core.windows.net/f40f3e69-38c0-43a9-9c99-2dde95a7fb20-9ecb/56ad02.jpeg?sv=2019-07-07&st=2021-07-29T10%3A10%3A27Z&se=2021-09-04T10%3A25%3A27Z&sr=b&sp=r&sig=G0Cz9sdxJZXyAbZpax0VHOJPfCFmszBWDhZGYBlM%2BdM%3D']
#a =  detect(img_url)
#print(a)
