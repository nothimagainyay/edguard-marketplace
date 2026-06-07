from detection.views import analyse_listing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Test dataset - known legitimate listings
legitimate_listings = [
    ("3 Bedroom Flat", "Well maintained apartment in a quiet estate. Modern kitchen and bathroom. Available immediately. Call for viewing.", 2500000, "Victoria Island Lagos"),
    ("2 Bedroom Apartment", "Spacious and clean apartment with good ventilation. Close to schools and markets. Reliable water supply.", 1800000, "Ikeja Lagos"),
    ("4 Bedroom Duplex", "Beautiful duplex in a serene environment. Large compound with parking space. Good road network.", 5000000, "Lekki Phase 1"),
    ("Studio Apartment", "Cozy studio apartment perfect for a single person. All facilities included. Safe neighborhood.", 800000, "Surulere Lagos"),
    ("5 Bedroom Mansion", "Luxury mansion with swimming pool and gym. Top quality finishing. 24 hour security.", 15000000, "Banana Island"),
]

# Test dataset - known fraudulent listings
fraudulent_listings = [
    ("Urgent Property Sale", "Act now!! Wire transfer payment only. God bless you. Advance fee required. Guaranteed deal overseas diplomat inheritance limited time!!!", 500, "Foreign"),
    ("Amazing House Giveaway", "Contact immediately urgent transfer fee advance payment western union guaranteed no questions asked overseas diplomat!!!!", 100, "Unknown"),
    ("Luxury Home Cheap", "Urgent urgent!!! Wire money immediately advance fee required. Limited time guaranteed deal. God bless. Foreign diplomat selling inheritance property!!!", 200, "Abroad"),
    ("Property For Sale Urgent", "Act now guaranteed deal wire transfer advance payment immediately no questions asked diplomat overseas inheritance blessing!!!", 300, "Overseas"),
    ("Cheap Mansion Available", "Urgently need to sell advance fee wire transfer immediately guaranteed blessing diplomat foreign inheritance contact now limited!!!", 150, "Foreign Location"),
]

print("=" * 60)
print("FYP FRAUD DETECTION SYSTEM - PERFORMANCE EVALUATION")
print("=" * 60)

y_true = []
y_pred = []
results = []

print("\nTesting LEGITIMATE listings...")
for title, desc, price, location in legitimate_listings:
    result = analyse_listing(title, desc, price, location)
    predicted = 1 if result['status'] == 'flagged' else 0
    y_true.append(0)
    y_pred.append(predicted)
    results.append((title, result['fraud_score'], result['status'], 'LEGITIMATE'))
    print(f"  '{title}' → Score: {result['fraud_score']} → {result['status']}")

print("\nTesting FRAUDULENT listings...")
for title, desc, price, location in fraudulent_listings:
    result = analyse_listing(title, desc, price, location)
    predicted = 1 if result['status'] == 'flagged' else 0
    y_true.append(1)
    y_pred.append(predicted)
    results.append((title, result['fraud_score'], result['status'], 'FRAUDULENT'))
    print(f"  '{title}' → Score: {result['fraud_score']} → {result['status']}")

print("\n" + "=" * 60)
print("PERFORMANCE METRICS")
print("=" * 60)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("=" * 60)
