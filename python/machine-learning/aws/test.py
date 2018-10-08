import boto3
import json
'''
client=boto3.client('rekognition')


fileName='test001.jpg'
bucket='smt1234'

response = client.detect_labels(Image={'S3Object':{'Bucket':bucket,'Name':fileName}})

print('Detected labels for ' + fileName)
    for label in response['Labels']:
        print (label['Name'] + ' : ' + str(label['Confidence']))
'''



imageFile = '/Users/msun/Documents/image/test005.jpg'
client = boto3.client('rekognition')

with open(imageFile, 'rb') as image:
    response = client.detect_labels(Image={'Bytes': image.read()})

print('Detected labels in ' + imageFile)
for label in response['Labels']:
    print(label['Name'] + ' : ' + str(label['Confidence']))

print('Done...')


with open(imageFile, 'rb') as image:
    response = client.detect_faces(Image={'Bytes': image.read()}, Attributes=["ALL"])

print('Detected faces for ' + imageFile)
for faceDetail in response['FaceDetails']:
    print('The detected face is between ' + str(faceDetail['AgeRange']['Low']) + ' and ' + str(faceDetail['AgeRange']['High']) + ' years old')
    print('Here are the other attributes:')
    print(json.dumps(faceDetail, indent=4, sort_keys=True))



import multiprocessing

multiprocessing.cpu_count()