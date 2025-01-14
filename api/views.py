# api/views.py
import os

from django.http import FileResponse
from omr_model_files.model_extraction_and_prediction import extract_staves, predict
from django.conf import settings
from omr_model_files.semantic_to_midi import semantic_to_midi, remove_extra_metadata
from omr_model_files.midi_to_mei import convert_midi_to_mei_with_musescore, merge_mei_files, merge_mei_batches
from omr_model_files.midi_left_and_right_combiner import merge_midi_files, merge_multiple_midi_files
from rest_framework.decorators import api_view, action
from rest_framework.response import Response
from rest_framework import viewsets, status
from .models import Upload, Profile
from .serializers import UploadSerializer, ProfileSerializer, UploadHistorySerializer
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.views import APIView
from django.contrib.auth.models import User
from rest_framework.authtoken.views import ObtainAuthToken
from django.contrib.auth import authenticate
from rest_framework.authtoken.models import Token
from django.http import JsonResponse
from django.middleware.csrf import get_token


# ViewSet for managing uploads
class UploadViewSet(viewsets.ModelViewSet):
    queryset = Upload.objects.all()
    serializer_class = UploadSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        # Only return uploads for the logged-in user
        return self.queryset.filter(user=self.request.user)

    def perform_create(self, serializer):
        print("Request user:", self.request.user)
        print("Is authenticated:", self.request.user.is_authenticated)
        user = self.request.user if self.request.user.is_authenticated else None
        serializer.save(user=user)

    @action(detail=False, methods=["post"])
    def batch_convert(self, request):
        """
        Process multiple uploaded files, convert them, and merge their final MIDI files.
        """
        file_ids = request.data.get("file_ids", [])
        tempo = float(request.data.get("tempo", 120))  # Default tempo
        time_signature = request.data.get("timeSignature", "4/4")  # Default time signature

        if not file_ids:
            return Response({"error": "No file IDs provided."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            final_midi_files = []
            final_mei_files = []
            first_file_id = file_ids[0]
            for file_id in file_ids:
                # Get the upload object
                upload = Upload.objects.get(id=file_id)
                if upload.status != "pending":
                    continue  # Skip files that are already processed or in progress

                upload.status = "processing"
                upload.save()

                try:
                    # Extract staves and process the file using existing logic
                    file_path = upload.file.path
                    output_dir = os.path.join(settings.MEDIA_ROOT, "extracted_staves")
                    final_midi_dir = os.path.join(settings.MEDIA_ROOT, "converted")
                    final_mei_dir = os.path.join(settings.MEDIA_ROOT, "mei")
                    os.makedirs(final_midi_dir, exist_ok=True)

                    staves = extract_staves(file_path, output_dir)
                    model_path = "omr_model_files/Camera-PrIMuS_hybrid_semantic_v1-10-10.meta"
                    voc_path = "omr_model_files/vocabulary_semantic.txt"

                    left_predictions = []
                    right_predictions = []
                    for i in range(len(staves)):
                        left, right = staves[i]
                        left_predictions.extend(predict(left, model_path, voc_path))
                        if left_predictions[-1] != "barline":
                            left_predictions.append("barline")
                        right_predictions.extend(predict(right, model_path, voc_path))
                        if right_predictions[-1] != "barline":
                            right_predictions.append("barline")

                    print(f"Left prediction:{left_predictions}\n Right prediction:{right_predictions}\n")

                    cleaned_left_predictions = remove_extra_metadata(left_predictions)
                    cleaned_right_predictions = remove_extra_metadata(right_predictions)

                    left_midi_file = f"{final_midi_dir}/left_{upload.id}.mid"
                    right_midi_file = f"{final_midi_dir}/right_{upload.id}.mid"
                    semantic_to_midi(cleaned_left_predictions, left_midi_file, tempo, time_signature)
                    semantic_to_midi(cleaned_right_predictions, right_midi_file, tempo, time_signature)

                    # Merge left and right MIDI files for this upload
                    final_midi_file = os.path.join(final_midi_dir, f"merged_{upload.id}.mid")
                    merge_midi_files(left_midi_file, right_midi_file, final_midi_file)
                    final_midi_files.append(final_midi_file)

                    left_mei_file = f"{final_mei_dir}/left_{upload.id}.mei"
                    right_mei_file = f"{final_mei_dir}/right_{upload.id}.mei"
                    convert_midi_to_mei_with_musescore(left_midi_file, left_mei_file)
                    convert_midi_to_mei_with_musescore(right_midi_file, right_mei_file)

                    merged_mei_file = f"{final_mei_dir}/merged_{upload.id}.mei"
                    merge_mei_files(left_mei_file, right_mei_file, time_signature, merged_mei_file)
                    final_mei_files.append(merged_mei_file)

                    # Update upload status and files
                    upload.converted_file.name = os.path.relpath(final_midi_file, settings.MEDIA_ROOT)
                    upload.converted_mei_file.name = os.path.relpath(merged_mei_file, settings.MEDIA_ROOT)
                    print(upload.converted_file.name)
                    print(upload.converted_mei_file.name)
                    upload.status = "completed"
                    upload.save()

                except Exception as e:
                    upload.status = "pending"  # Reset status on failure
                    upload.save()
                    return Response({"error": f"Error processing file {file_id}: {str(e)}"},
                                    status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Merge all final MIDI files into a single file
            merged_midi_path = os.path.join(settings.MEDIA_ROOT, "converted", f"merged_batch_{first_file_id}.mid")
            merge_multiple_midi_files(final_midi_files, merged_midi_path)
            #
            # merged_mei_path = os.path.join(settings.MEDIA_ROOT, "mei", f"merged_{first_file_id}.mei")
            # merge_mei_batches(final_mei_files, merged_mei_path)

            upload = Upload.objects.get(id=file_ids[0])
            upload.converted_file.name = os.path.relpath(merged_midi_path, settings.MEDIA_ROOT)
            upload.converted_mei_file.name = os.path.relpath(final_mei_files[0], settings.MEDIA_ROOT)
            print(upload.converted_file.name)
            upload.status = "completed"
            upload.save()

            return Response({"status": "Batch conversion completed", "merged_file_url": merged_midi_path},
                            status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # @action(detail=True, methods=["post"])
    # def convert(self, request, pk=None):
    #     upload = self.get_object()
    #     if upload.status != "pending":
    #         return Response({"error": "File is already processed or in progress."}, status=status.HTTP_400_BAD_REQUEST)
    #
    #     tempo = float(request.data.get("tempo"))
    #     time_signature = request.data.get("timeSignature")
    #
    #     # Update status to 'processing'
    #     upload.status = "processing"
    #     upload.save()
    #
    #     # Extract staves
    #     try:
    #         file_path = upload.file.path
    #         output_dir = os.path.join(settings.MEDIA_ROOT, "extracted_staves")
    #         final_midi_dir = os.path.join(settings.MEDIA_ROOT, "converted")
    #         final_mei_dir = os.path.join(settings.MEDIA_ROOT, "mei")
    #         os.makedirs(final_midi_dir, exist_ok=True)
    #         staves = extract_staves(file_path, output_dir)
    #         print(staves)
    #         # Run predictions on extracted staves
    #         model_path = "omr_model_files/Camera-PrIMuS_hybrid_semantic_v1-10-10.meta"
    #         voc_path = "omr_model_files/vocabulary_semantic.txt"
    #         midi_files = []
    #         left_predictions = []
    #         right_predictions = []
    #         for i in range(len(staves)):
    #             left, right = staves[i]
    #             left_predictions.extend(predict(left, model_path, voc_path))
    #             left_predictions.append("barline")
    #             right_predictions.extend(predict(right, model_path, voc_path))
    #             right_predictions.append("barline")
    #
    #         print(f"Left prediction:{left_predictions}\n Right prediction:{right_predictions}\n")
    #
    #         cleaned_left_predictions = remove_extra_metadata(left_predictions)
    #         cleaned_right_predictions = remove_extra_metadata(right_predictions)
    #
    #         print(f"Left prediction:{cleaned_left_predictions}\n Right prediction:{cleaned_right_predictions}\n")
    #
    #         left_midi_file = f"{final_midi_dir}/left_{upload.id}.mid"
    #         right_midi_file = f"{final_midi_dir}/right_{upload.id}.mid"
    #         semantic_to_midi(cleaned_left_predictions, left_midi_file, tempo, time_signature)
    #         semantic_to_midi(cleaned_right_predictions, right_midi_file, tempo, time_signature)
    #
    #         left_mei_file = f"{final_mei_dir}/left_{upload.id}.mei"
    #         right_mei_file = f"{final_mei_dir}/right_{upload.id}.mei"
    #         convert_midi_to_mei_with_musescore(left_midi_file,left_mei_file)
    #         convert_midi_to_mei_with_musescore(right_midi_file, right_mei_file)
    #
    #         merged_mei_file = f"{final_mei_dir}/merged_{upload.id}.mei"
    #         merge_mei_files(left_mei_file, right_mei_file, time_signature, merged_mei_file)
    #
    #         # Merge all MIDI pairs into a single file
    #         final_midi_file = os.path.join(final_midi_dir, f"merged_{upload.id}.mid")
    #         merge_midi_files(left_midi_file, right_midi_file, final_midi_file)
    #
    #         # Update the Upload model instance
    #         upload.converted_file.name = os.path.relpath(final_midi_file, settings.MEDIA_ROOT)
    #         upload.converted_mei_file.name = os.path.relpath(merged_mei_file, settings.MEDIA_ROOT)
    #         print(upload.converted_file.name)
    #         print(upload.converted_mei_file.name)
    #         upload.status = "completed"
    #         upload.save()
    #
    #         return Response({"status": "Conversion completed"}, status=status.HTTP_200_OK)
    #     except Exception as e:
    #         upload.status = "pending"  # Reset status on failure
    #         upload.save()
    #         return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=["get"])
    def status(self, request, pk=None):
        # Return the current status of the file, Check the status of a file conversion.

        upload = self.get_object()
        return Response({"status": upload.status}, status=status.HTTP_200_OK)

    # @action(detail=True, methods=["get"])
    # def download(self, request, pk=None):
    #     # Return the converted file for download, Download a converted MIDI file.
    #     upload = self.get_object()
    #     if upload.converted_file:
    #         return Response({"file_url": upload.converted_file.url}, status=status.HTTP_200_OK)
    #     return Response({"error": "File not converted yet."}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=["get"])
    def mei(self, request, pk=None):
        # Retrieve the upload object
        upload = self.get_object()

        # Check if the converted MIDI file exists
        if upload.converted_mei_file and os.path.exists(upload.converted_mei_file.path):
            # Stream the file to the user
            return FileResponse(
                open(upload.converted_mei_file.path, "rb"),
                filename=os.path.basename(upload.converted_mei_file.path),  # Sets the file name for the download
            )

        # If the file doesn't exist, return an error response
        return Response({"error": "Converted MIDI file not found."}, status=status.HTTP_404_NOT_FOUND)

    @action(detail=True, methods=["get"])
    def download(self, request, pk=None):
        # Retrieve the upload object
        upload = self.get_object()

        # Check if the converted MIDI file exists
        if upload.converted_file and os.path.exists(upload.converted_file.path):
            # Stream the file to the user
            return FileResponse(
                open(upload.converted_file.path, "rb"),
                as_attachment=True,  # Forces download
                filename=os.path.basename(upload.converted_file.path),  # Sets the file name for the download
            )

        # If the file doesn't exist, return an error response
        return Response({"error": "Converted MIDI file not found."}, status=status.HTTP_404_NOT_FOUND)


# ViewSet for managing profiles
class ProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        # Dynamically create a profile if it doesn't exist
        profile, created = Profile.objects.get_or_create(user=request.user)
        serializer = ProfileSerializer(profile)
        return Response(serializer.data)

    def put(self, request):
        # Update user profile details.

        profile = Profile.objects.get(user=request.user)
        serializer = ProfileSerializer(profile, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)


class UploadHistoryView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        # Fetch all uploads for the logged-in user
        uploads = Upload.objects.filter(user=request.user)

        serializer = UploadHistorySerializer(uploads, many=True)
        return Response(serializer.data)


class RegisterUserView(APIView):
    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')
        email = request.data.get('email')

        if not username or not password or not email:
            return Response({'error': 'All fields are required.'}, status=status.HTTP_400_BAD_REQUEST)

        if User.objects.filter(username=username).exists():
            return Response({'error': 'Username already exists.'}, status=status.HTTP_400_BAD_REQUEST)

        user = User.objects.create_user(username=username, password=password, email=email)
        return Response({'message': 'User created successfully.'}, status=status.HTTP_201_CREATED)


class CustomObtainAuthToken(ObtainAuthToken):
    def post(self, request, *args, **kwargs):
        # Print the request data for debugging
        print("Request data:", request.data)
        response = super().post(request, *args, **kwargs)
        print("Response data:", response.data)
        return response


class LoginView(APIView):
    def post(self, request, *args, **kwargs):
        # Log the request data
        if request:
            print("Request received:", request.data)
        else:
            print("No request received")

        username = request.data.get('username')
        password = request.data.get('password')

        user = authenticate(request, username=username, password=password)
        if user:
            # Ensure the user is active
            if not user.is_active:
                return Response({"error": "User account is disabled."}, status=status.HTTP_401_UNAUTHORIZED)

            # Generate or retrieve token
            token, _ = Token.objects.get_or_create(user=user)

            # Return the token
            return Response({"token": token.key}, status=status.HTTP_200_OK)

        return Response({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)


def test_view(request):
    print("Request received:", request)
    return JsonResponse({"message": "Test endpoint reached"})


def get_csrf_token(request):
    return JsonResponse({'csrfToken': get_token(request)})