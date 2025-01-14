# api/models.py
from django.db import models
from django.contrib.auth.models import User

class Upload(models.Model):
    file = models.FileField(upload_to='upload/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(
        max_length=20, 
        choices=[("pending", "Pending"), ("processing", "Processing"), ("completed", "Completed")], 
        default="pending"  
    )
    converted_file = models.FileField(upload_to="converted/", null=True, blank=True)  # Holds converted MIDI files
    converted_mei_file = models.FileField(upload_to="mei/", null=True, blank=True)  # Holds converted MEI files
    user = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True
    )  
    def __str__(self):
        return f"Upload by {self.user.username} on {self.uploaded_at}: {self.file.name} - {self.status}"


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    email = models.EmailField()
    preferred_language = models.CharField(max_length=50, default='English')

    def __str__(self):
        return f"Profile of {self.user.username}"
