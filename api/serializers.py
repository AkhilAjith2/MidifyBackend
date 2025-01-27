# api/serializers.py

from rest_framework import serializers
from .models import Upload, Profile
from rest_framework.serializers import ModelSerializer
from django.contrib.auth.models import User


class UserSerializer(ModelSerializer):
    """Serializer for the User model"""
    class Meta:
        model = User
        fields = ['id', 'username', 'first_name', 'last_name', 'email']

class UploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Upload
        fields = ['id', 'file', 'uploaded_at', 'user', 'status', 'converted_file', 'converted_mei_file']  # Include relevant fields
        read_only_fields = ['id', 'uploaded_at', 'user']


class ProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = Profile
        fields = ['id', 'user', 'first_name', 'last_name', 'email', 'preferred_language']


class UploadHistorySerializer(serializers.ModelSerializer):
    date = serializers.DateTimeField(source='uploaded_at', format='%Y-%m-%d %H:%M:%S')  # Format date
    name = serializers.CharField(source='file.name')  # Get file name

    class Meta:
        model = Upload
        fields = ['id', 'name', 'date']  # Add fields you want to expose