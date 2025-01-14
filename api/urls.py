# api/urls.py

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import UploadViewSet, ProfileView, UploadHistoryView, RegisterUserView, CustomObtainAuthToken, LoginView, test_view, get_csrf_token, FetchImage, FetchMIDI
from rest_framework.authtoken.views import obtain_auth_token
from django.views.decorators.csrf import csrf_exempt

router = DefaultRouter()
router.register(r'upload', UploadViewSet, basename='upload')

urlpatterns = [
    path('', include(router.urls)),
    path('profile/', ProfileView.as_view(), name='profile'),
    path('history/', UploadHistoryView.as_view(), name='upload-history'),
    path('history/<int:upload_id>/', FetchImage.as_view(), name='upload-history-detail'),
    path('history/<int:upload_id>/fetch-midi/', FetchMIDI.as_view(), name='fetch_midi'),
    path('register/', RegisterUserView.as_view(), name='register'),
    path('auth/login/', csrf_exempt(LoginView.as_view()), name='custom_login'),
    path('test/', test_view),
    path('csrf-token/', get_csrf_token, name='csrf_token'),
]