from django.contrib import admin
from .models import Listing

@admin.register(Listing)
class ListingAdmin(admin.ModelAdmin):
    list_display = ['title', 'owner', 'price', 'location', 'status', 'fraud_score', 'created_at']
    list_filter = ['status']
    search_fields = ['title', 'description', 'location']
    readonly_fields = ['fraud_score', 'created_at']
    
    actions = ['approve_listings', 'flag_listings']
    
    def approve_listings(self, request, queryset):
        queryset.update(status='approved')
        self.message_user(request, "Selected listings have been approved.")
    approve_listings.short_description = "Approve selected listings"
    
    def flag_listings(self, request, queryset):
        queryset.update(status='flagged')
        self.message_user(request, "Selected listings have been flagged.")
    flag_listings.short_description = "Flag selected listings as fraudulent"