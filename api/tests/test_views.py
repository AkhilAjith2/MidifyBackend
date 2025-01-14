from rest_framework.test import APITestCase
from rest_framework import status
from django.contrib.auth.models import User
from api.models import Upload
from rest_framework.authtoken.models import Token


class UploadViewSetTest(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="testuser", password="testpassword")
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {self.user.auth_token.key}')
        self.client.login(username="testuser", password="testpassword")
        self.upload = Upload.objects.create(
            file="test_file.txt", 
            user=self.user, 
            status="pending"
        )

    def test_upload_list(self):
        response = self.client.get('/api/upload/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)

    def test_upload_create(self):
        with open('test_file.txt', 'w') as f:
            f.write("Test content")
        with open('test_file.txt', 'rb') as f:
            response = self.client.post('/api/upload/', {'file': f}, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

    def test_upload_status(self):
        response = self.client.get(f'/api/upload/{self.upload.id}/status/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], 'pending')

    def test_upload_convert(self):
        response = self.client.post(f'/api/upload/{self.upload.id}/convert/')
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        self.upload.refresh_from_db()
        self.assertEqual(self.upload.status, 'processing')

    def test_upload_download_error(self):
        response = self.client.get(f'/api/upload/{self.upload.id}/download/')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.data['error'], "File not converted yet.")

    def test_upload_download_success(self):
        # Test download when converted_file is present
        self.upload.converted_file = "converted/test_converted_file.mid"
        self.upload.save()

        response = self.client.get(f'/api/upload/{self.upload.id}/download/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('file_url', response.data)
        self.assertEqual(response.data['file_url'], self.upload.converted_file.url)
        
class ProfileViewTest(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="testuser", password="testpassword")
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {self.user.auth_token.key}')
        self.client.login(username="testuser", password="testpassword")

    def test_get_profile(self):
        response = self.client.get('/api/profile/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['preferred_language'], 'English')

    def test_update_profile(self):
        data = {'first_name': 'John', 'last_name': 'Doe', 'preferred_language': 'Spanish'}
        response = self.client.put('/api/profile/', data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['preferred_language'], 'Spanish')
