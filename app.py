from scenera.node import SceneMark
from emotion_with_url import *
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

## The following environment variables are assigned to you by the Developer Portal
ANALYSIS_ID = "emotion_recognition_algorithm_1"
ANALYSIS_DESCRIPTION = "Detect emotion of faces through Azure API"
NODE_ID = "EmotionNode"
EVENT_TYPE = "ItemPresence"

#CHARMI: No need to specify a custom event type, because you already have an event type: ItemPresence
CUSTOM_EVENT_TYPE = ""
#NodeSequencerAddress = "https://node-sdk-demo.cognitiveservices.azure.com"

DEBUG = True

## These environment variables relate to the face recognition algorithm employed by this Node
#CHARMI: Do you need this DB as well for the Emotions?
FACE_DB_URI = "mongodb://face-recognition-1:mgT6eNYp88f25HL4JogVsH20KznhvuhKg92aJzpK9lNB3A1RVhiKWA0MVwXDRFewd2hIVNXN4ECnnSan4CB7iA==@face-recognition-1.mongo.cosmos.azure.com:10255/?ssl=true&replicaSet=globaldb&retrywrites=false&maxIdleTimeMS=120000&appName=@face-recognition-1@"
FACE_CLIENT_ENDPOINT = "https://facedemonstration.cognitiveservices.azure.com/"
FACE_CLIENT_KEY = "055cc8b8d56c42e5bddc8e8a315c080d"

## The Node ID assigned by the Developer Portal is used as a URI
@app.route('/SceneMark',methods=['POST'])
def node_endpoint(test = False):
    """
    A Flask implementation of the Scenera Node SDK

    Parameters
    ----------
    request : incoming request
        The incoming request including a SceneMark and a NodeSequencerAddress

    Returns
    -------
        Returns a SceneMark to the Data Pipeline - Node Sequencer - using the return_scenemark_to_ns
        method
    """

    ## This first thing we do is load the request into the SceneMark object,
    ## and load the information from the Developer Portal
    scenemark = SceneMark(
        request = request,
        node_id = NODE_ID,
        event_type = EVENT_TYPE,
        custom_event_type = CUSTOM_EVENT_TYPE,
        analysis_id = ANALYSIS_ID,
        analysis_description = ANALYSIS_DESCRIPTION)

    ## For testing purposes, log what is coming in.
    scenemark.save_request("SM", name = "scenemark_received")

    ## Get a uri dict, returns, for example:
    # #{'Thumbnail': https://scenedatauri.example.com/1234_thumb.jpg,
    # #'RGBStill': https://scenedatauri.example.com/1234_still.jpg,
    # #'RGBVideo': ''}
    scenedata_uris = scenemark.get_scenedata_datatype_uri_dict()

    ## Run face detection on the still image (the second in the list)
    ## This refers to the file found in face_recognition.py and contains
    ## essentially the body of this Node's algorithm

    result = run_emotion_recognition(
        scenedata_uris['RGBStill'],
        FACE_DB_URI,
        FACE_CLIENT_ENDPOINT,
        FACE_CLIENT_KEY)

    print("\n........................\nResult: ", result, "\n........................")

    ## If the ProcessingStatus returns anything but 'Recognized', we log
    ## what went wrong
    if result['ProcessingStatus'] != "Detected":
        print("\nError\n", result['Error_Message'])
        scenemark.add_analysis_list_item(
            processing_status = result['ProcessingStatus'],
            error_message = str(result['Error_Message'])
        )

    else:
        attributes = []
        try:
            attributes.append(
                scenemark.generate_attribute_item(
                    attribute = 'age',
                    value = str(result['Values']['age']),
                )
            )
        except:
            print("No age feature!")

        try:
            attributes.append(
                scenemark.generate_attribute_item(
                    attribute = 'glasses',
                    value = str(result['Values']['glasses'])
                    )
                )
        except:
            print("No glasses feature!")

        try:
            for emotion in result['Values']['emotion']:
                if result['Values']['emotion'][emotion]:
                    attributes.append(
                        scenemark.generate_attribute_item(
                            attribute = 'emotion',
                            value = emotion
                        )
                    )
        except:
            print("No emotion present!")

        ## We add detected objects for each object that we find within the image
        detected_objects = []
        detected_objects.append(
            scenemark.generate_detected_object_item(
                nice_item_type = "Face",
                item_type_count = 1,
                related_scenedata_id = scenemark.get_id_from_uri(scenedata_uris['RGBStill']),
                attributes = attributes
                ))

        ## When we have the detected objects, we log them in the top level of the analysis list
        scenemark.add_analysis_list_item(
            processing_status = result['ProcessingStatus'],
            total_item_count = len(detected_objects),
            detected_objects = detected_objects,
            )

        scenemark.save_request("SM", name = "saved_scenemark")
        print("############################################## FINAL SCENEMARKS")
        #exit()

    ## For testing purposes we save the outgoing SceneMark, such that we can
    ## inspect it in case any errors occur
    # scenemark.save_request("SM", "scenemark_to_be_sent")

    ## We automatically return the SceneMark back to the NodeSequencer
    scenemark.return_scenemark_to_ns(test)
    return "Success"

## This endpoint returns the SceneMark back to the sender, rather than the NodeSequencer
## for example for testing through Postman
@app.route(f'/{NODE_ID}/test',methods=['POST'])
def test_endpoint():
    return node_endpoint(test = True)

## Use this to check whether the Node is live
@app.route(f'/{NODE_ID}/health')
def health_endpoint():
    return jsonify({"status": "UP"}), 200

if __name__ == "__main__":
    app.run(
      host="127.0.0.1",
      port=5000,
      debug=DEBUG
      )
    ## TODO: Change this to reflect the route
