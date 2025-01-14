from django.test import TestCase
from django.contrib.auth.models import User
from api.models import Upload, Profile

class UploadModelTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="testuser", password="testpassword")
        self.upload = Upload.objects.create(
            file="test_file.txt", 
            user=self.user, 
            status="pending"
        )

    def test_upload_creation(self):
        self.assertEqual(self.upload.file.name, "test_file.txt")
        self.assertEqual(self.upload.status, "pending")
        self.assertEqual(self.upload.user.username, "testuser")

    def test_upload_str(self):
        self.assertEqual(
            str(self.upload),
            f"Upload by {self.user.username} on {self.upload.uploaded_at}: {self.upload.file.name} - {self.upload.status}"
        )

class ProfileModelTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="testuser", password="testpassword")
        self.profile = Profile.objects.get(user=self.user)

    def test_profile_creation(self):
        self.assertEqual(self.profile.user.username, "testuser")
        self.assertEqual(self.profile.preferred_language, "English")
