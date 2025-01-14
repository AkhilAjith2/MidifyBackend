# from django.test import TestCase
# from django.test import TestCase
# from rest_framework.test import APIClient
# from rest_framework import status
# from django.contrib.auth.models import User
# from .models import Upload


# # Create your tests here.
# class BaseTestCase(TestCase):
#     def setUp(self):
#         # Create a test user
#         self.user = User.objects.create_user(username='testuser', password='testpassword')
#         self.client = APIClient()
#         # Authenticate the test client
#         self.client.force_authenticate(user=self.user)

# class UploadTestCase(BaseTestCase):
#     def test_upload_file(self):
#         # Create a mock file
#         with open('testfile.txt', 'w') as f:
#             f.write('This is a test file.')
        
#         with open('testfile.txt', 'rb') as f:
#             response = self.client.post('/api/upload/', {'file': f}, format='multipart')
        
#         # Assertions
#         self.assertEqual(response.status_code, status.HTTP_201_CREATED)
#         self.assertIn('id', response.data)
#         self.assertIn('file', response.data)

#     def test_upload_file_unauthenticated(self):
#         unauthenticated_client = APIClient()
#         with open('testfile.txt', 'rb') as f:
#             response = unauthenticated_client.post('/api/upload/', {'file': f}, format='multipart')
        
#         # Assertions
#         self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)


# class ProfileTestCase(BaseTestCase):
#     def test_retrieve_profile(self):
#         response = self.client.get(f'/api/profile/{self.user.id}/')
#         self.assertEqual(response.status_code, status.HTTP_200_OK)
#         self.assertEqual(response.data['username'], self.user.username)

#     def test_update_profile(self):
#         data = {'first_name': 'Updated', 'last_name': 'User'}
#         response = self.client.put(f'/api/profile/{self.user.id}/', data)
#         self.assertEqual(response.status_code, status.HTTP_200_OK)
#         self.assertEqual(response.data['first_name'], 'Updated')
#         self.assertEqual(response.data['last_name'], 'User')

# class HistoryTestCase(BaseTestCase):
#     def setUp(self):
#         super().setUp()
#         # Create some mock uploads
#         Upload.objects.create(user=self.user, file='testfile1.mid', status='completed')
#         Upload.objects.create(user=self.user, file='testfile2.mid', status='pending')

#     def test_retrieve_history(self):
#         response = self.client.get('/api/upload/history/')
#         self.assertEqual(response.status_code, status.HTTP_200_OK)
#         self.assertEqual(len(response.data), 2)  # Two uploads for the user
#         self.assertEqual(response.data[0]['status'], 'completed')
#         self.assertEqual(response.data[1]['status'], 'pending')
