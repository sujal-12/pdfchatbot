from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    # Main page URL: /
    path('', views.index, name='index'), 
    
    # API endpoint for file upload: /upload/
    path('upload/', views.upload_pdf_view, name='upload_pdf'), 
    
    # API endpoint for sending chat messages: /chat/
    path('chat/', views.chat_view, name='chat'), 
    
    # API endpoint for cleanup on exit: /cleanup/
    path('cleanup/', views.cleanup_view, name='cleanup'),
]
