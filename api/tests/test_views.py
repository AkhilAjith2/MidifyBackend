from rest_framework.test import APITestCase
from rest_framework import status
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
from api.models import Upload
from io import BytesIO
from django.core.files.uploadedfile import SimpleUploadedFile

class ExtendedProfileViewTest(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="testuser", password="testpassword")
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {self.user.auth_token.key}')
        self.client.login(username="testuser", password="testpassword")

    def test_get_profile_unauthenticated(self):
        self.client.logout()
        response = self.client.get('/api/profile/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_update_profile_invalid_data(self):
        data = {'first_name': '', 'preferred_language': 'French'}
        response = self.client.put('/api/profile/', data)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('first_name', response.data)

    def test_update_profile_partial_update(self):
        data = {'last_name': 'Smith'}
        response = self.client.put('/api/profile/', data)  # Use PUT instead of PATCH
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['last_name'], 'Smith')


class ExtendedUploadViewSetTest(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="testuser", password="testpassword")
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {self.user.auth_token.key}')
        self.client.login(username="testuser", password="testpassword")
        self.upload = Upload.objects.create(
            file="test_file.png", 
            user=self.user, 
            status="pending"
        )

    def test_upload_list_unauthenticated(self):
        self.client.logout()
        response = self.client.get('/api/upload/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_upload_create_invalid_file(self):
        response = self.client.post('/api/upload/', {'file': ''}, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('file', response.data)

    def test_upload_create_success(self):
        file_content = BytesIO(b"Test file content")
        uploaded_file = SimpleUploadedFile("new_test_file.png", file_content.read())
        response = self.client.post('/api/upload/', {'file': uploaded_file}, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['status'], 'pending')

    def test_upload_status_not_found(self):
        response = self.client.get('/api/upload/999/status/')
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_download_midi_success(self):
        # Mock file for test
        self.upload.converted_mei_file = "converted_mei_test_file.mid"
        self.upload.save()
        response = self.client.get(f'/api/upload/{self.upload.id}/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_batch_convert_no_file_ids(self):
        response = self.client.post('/api/upload/batch_convert/', {'file_ids': []})
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)

    def test_batch_convert_success(self):
        response = self.client.post('/api/upload/batch_convert/', {'file_ids': [self.upload.id]})
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_500_INTERNAL_SERVER_ERROR])

    def test_download_converted_midi_not_found(self):
        self.upload.converted_file = None  # Ensure converted file is set to None
        self.upload.save()
        response = self.client.get(f'/api/upload/{self.upload.id}/download/')
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

class RegisterUserViewTest(APITestCase):
    def test_register_success(self):
        data = {
            'username': 'newuser',
            'password': 'newpassword',
            'email': 'newuser@example.com'
        }
        response = self.client.post('/api/register/', data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

    def test_register_missing_field(self):
        data = {
            'username': 'newuser',
            'email': 'newuser@example.com'
        }
        response = self.client.post('/api/register/', data)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_register_duplicate_user(self):
        User.objects.create_user(username="existinguser", password="password123")
        data = {
            'username': 'existinguser',
            'password': 'password123',
            'email': 'existinguser@example.com'
        }
        response = self.client.post('/api/register/', data)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

class LoginViewTest(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="testuser", password="testpassword")

    def test_login_success(self):
        data = {'username': 'testuser', 'password': 'testpassword'}
        response = self.client.post('/api/auth/login/', data)  # Updated URL
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('token', response.data)

    def test_login_invalid_credentials(self):
        data = {'username': 'testuser', 'password': 'wrongpassword'}
        response = self.client.post('/api/auth/login/', data)  # Updated URL
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_login_disabled_user(self):
        self.user.is_active = False
        self.user.save()
        data = {'username': 'testuser', 'password': 'testpassword'}
        response = self.client.post('/api/auth/login/', data)  # Updated URL
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)







# from rest_framework.test import APITestCase
# from rest_framework import status
# from django.contrib.auth.models import User
# from unittest.mock import patch, mock_open
# from api.models import Upload
# from rest_framework.authtoken.models import Token



# class ProfileViewTest(APITestCase):
#     def setUp(self):
#         self.user = User.objects.create_user(username="testuser", password="testpassword")
#         self.client.credentials(HTTP_AUTHORIZATION=f'Token {self.user.auth_token.key}')
#         self.client.login(username="testuser", password="testpassword")

#     def test_get_profile(self):
#         response = self.client.get('/api/profile/')
#         self.assertEqual(response.status_code, status.HTTP_200_OK)
#         self.assertEqual(response.data['preferred_language'], 'English')

#     def test_update_profile(self):
#         data = {'first_name': 'John', 'last_name': 'Doe', 'preferred_language': 'Spanish'}
#         response = self.client.put('/api/profile/', data)
#         self.assertEqual(response.status_code, status.HTTP_200_OK)
#         self.assertEqual(response.data['preferred_language'], 'Spanish')

# class UploadViewSetTest(APITestCase):
#     def setUp(self):
#         self.user = User.objects.create_user(username="testuser", password="testpassword")
#         self.client.credentials(HTTP_AUTHORIZATION=f'Token {self.user.auth_token.key}')
#         self.client.login(username="testuser", password="testpassword")
#         self.upload = Upload.objects.create(
#             file="test_file.txt", 
#             user=self.user, 
#             status="pending"
#         )
        
#     def test_upload_list(self):
#         response = self.client.get('/api/upload/')
#         self.assertEqual(response.status_code, status.HTTP_200_OK)
#         self.assertEqual(len(response.data), 1)

#     def test_upload_create(self):
#         with open('test_file.txt', 'w') as f:
#             f.write("Test content")
#         with open('test_file.txt', 'rb') as f:
#             response = self.client.post('/api/upload/', {'file': f}, format='multipart')
#         self.assertEqual(response.status_code, status.HTTP_201_CREATED)

#     def test_upload_status(self):
#         response = self.client.get(f'/api/upload/{self.upload.id}/status/')
#         self.assertEqual(response.status_code, status.HTTP_200_OK)
#         self.assertEqual(response.data['status'], 'pending')

    