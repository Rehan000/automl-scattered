import boto3


class Rekognition:
    def __init__(self):
        self.client = boto3.client('rekognition')
        self.bucket_face_images = []

    def create_collection(self, collection_id):
        """Creates a new collection of the given name if it doesn't exist."""
        try:
            print('Creating collection: ' + collection_id)
            response = self.client.create_collection(CollectionId=collection_id)
            print('Collection ARN: ' + response['CollectionArn'])
            print('Status code: ' + str(response['StatusCode']))
            print('Done...\n')
        except self.client.exceptions.ResourceAlreadyExistsException:
            print('Collection already exists.\n')

    def list_collections(self):
        """Lists all the collection names and total number of collections."""
        print('Displaying collections...')
        response = self.client.list_collections()
        collection_count = 0
        done = False

        while not done:
            collections = response['CollectionIds']

            for collection in collections:
                print("Collection name:", collection)
                collection_count += 1
            if 'NextToken' in response:
                next_token = response['NextToken']
                response = self.client.list_collections(NextToken=next_token)

            else:
                done = True

        print("Total collections:", collection_count)
        print()

    def add_faces_to_collection(self, bucket_name, collection_id):
        """Add all the faces from an S3 bucket into the collection."""
        bucket_object = boto3.resource('s3').Bucket(bucket_name)

        for file in bucket_object.objects.all():
            if file.key[-1] != "/":
                self.bucket_face_images.append(file.key)

        for image in self.bucket_face_images:
            response = self.client.index_faces(CollectionId=collection_id,
                                               Image={'S3Object': {'Bucket': bucket_name, 'Name': image}},
                                               ExternalImageId=image.split('/')[0],
                                               MaxFaces=1,
                                               QualityFilter="AUTO",
                                               DetectionAttributes=['ALL'])
            print("Face added to collection")
        print()

    def list_faces_in_collection(self, collection_id):
        """Lists all the faces in a given collection."""
        faces_count = 0
        tokens = True
        response = self.client.list_faces(CollectionId=collection_id)
        print('Faces in collection: ' + collection_id)
        while tokens:
            faces = response['Faces']
            for face in faces:
                print(face)
                faces_count += 1
            if 'NextToken' in response:
                next_token = response['NextToken']
                response = self.client.list_faces(CollectionId=collection_id, NextToken=next_token)
            else:
                tokens = False
        print("Total faces:", faces_count)
        print()

    def search_face(self, collection_id, source_image):
        """Search a face from the given collection, given a source image."""
        with open(source_image, 'rb') as img_file:
            image = {'Bytes': img_file.read()}

        response = self.client.search_faces_by_image(CollectionId=collection_id, Image=image)
        return response
