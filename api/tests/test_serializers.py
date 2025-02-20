from django.test import TestCase
from api.serializers import UploadSerializer, ProfileSerializer
from api.models import Upload, Profile
from django.contrib.auth.models import User

class UploadSerializerTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="testuser", password="testpassword")
        self.upload = Upload.objects.create(
            file="test_file.txt", 
            user=self.user, 
            status="pending"
        )

    def test_upload_serializer(self):
        serializer = UploadSerializer(instance=self.upload)
        expected_data = {
            'id': self.upload.id,
            'user': self.user.id,
            'file': f'/media/{self.upload.file.name}',  # Ensure correct format if using media URLs
            'uploaded_at': serializer.data['uploaded_at'],  # Use dynamic datetime format from serializer
            'status': 'pending',
            'converted_file': None,  # Expected to be None for initial uploads
            'converted_mei_file': None,  # Make sure to use the correct field name as per serializer
        }
        self.assertEqual(serializer.data, expected_data)


class ProfileSerializerTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="testuser", password="testpassword")
        self.profile = Profile.objects.get(user=self.user)

    def test_profile_serializer(self):
        serializer = ProfileSerializer(instance=self.profile)
        expected_data = {
            'id': self.profile.id,
            'user': self.user.id,
            'first_name': '',
            'last_name': '',
            'email': '',
            'preferred_language': 'English',
        }
        self.assertEqual(serializer.data, expected_data)
