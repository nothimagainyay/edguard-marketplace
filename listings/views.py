from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from .models import Listing
from .forms import ListingForm
from detection.views import analyse_listing

def listing_list(request):
    listings = Listing.objects.filter(status='approved')
    return render(request, 'listings/listing_list.html', {'listings': listings})

def listing_detail(request, pk):
    listing = get_object_or_404(Listing, pk=pk)
    return render(request, 'listings/listing_detail.html', {'listing': listing})

@login_required
def listing_create(request):
    if request.method == 'POST':
        form = ListingForm(request.POST, request.FILES)
        if form.is_valid():
            listing = form.save(commit=False)
            listing.owner = request.user
            
            # Run fraud detection
            result = analyse_listing(
                title=listing.title,
                description=listing.description,
                price=listing.price,
                location=listing.location
            )
            
            listing.fraud_score = result['fraud_score']
            listing.status = result['status']
            listing.save()
            
            return redirect('listing_detail', pk=listing.pk)
    else:
        form = ListingForm()
    return render(request, 'listings/listing_create.html', {'form': form})