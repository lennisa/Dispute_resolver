import json
import random

def generate_synthetic_disputes(num_cases=300):
    reasons = [
        "item_not_received", 
        "fraudulent_transaction", 
        "defective_item", 
        "duplicate_charge",
        "subscription_cancellation", 
        "unrecognized_merchant",
        "wrong_item_shipped",
        "returned_item_no_refund",
        "service_not_rendered",
        "price_discrepancy"
    ]
    labels = ["card member wins", "merchant wins", "partial"]
    
    dataset = []
    
    for i in range(1, num_cases + 1):
        reason = random.choice(reasons)
        label = random.choice(labels)
        
        if reason == "item_not_received":
            cust_stmt = f"I never received order #{1000 + i}. Tracking shows delivered to a wrong address or zip code."
            merch_stmt = f"Package was successfully delivered to the exact shipping address provided at checkout for order #{1000 + i}."
            cust_ev = ["tracking_screenshot.png"]
            merch_ev = ["carrier_delivery_confirmation.pdf", "signature_log.png"]
            
        elif reason == "fraudulent_transaction":
            cust_stmt = f"I did not authorize transaction txn_{5000 + i} for ${random.randint(50, 500)}.00 on my card."
            merch_stmt = f"Transaction txn_{5000 + i} was processed securely with 3DS validation, correct billing zip, and matching CVV."
            cust_ev = ["fraud_affidavit.pdf"]
            merch_ev = ["auth_logs.json", "ip_address_match.png"]
            
        elif reason == "defective_item":
            cust_stmt = f"The product for order #{2000 + i} arrived completely broken, damaged, and entirely unusable."
            merch_stmt = f"Item shipped was brand new in box and quality tested. Customer failed to return the item within our 30-day policy."
            cust_ev = ["damage_photo.png"]
            merch_ev = ["return_policy.pdf", "qc_inspection_sheet.pdf"]
            
        elif reason == "duplicate_charge":
            cust_stmt = f"I was charged twice by mistake for the exact same purchase on invoice #{3000 + i}."
            merch_stmt = f"The two charges for invoice #{3000 + i} represent two separate, distinct line-item orders placed minutes apart."
            cust_ev = ["bank_statement_snippet.png"]
            merch_ev = ["itemized_receipt.pdf", "pos_terminal_log.pdf"]
            
        elif reason == "subscription_cancellation":
            cust_stmt = f"I cancelled my recurring monthly membership months ago, but subscription fee #{4000 + i} was still billed to my account."
            merch_stmt = f"No cancellation request was ever received through our portal before billing cycle #{4000 + i} occurred."
            cust_ev = ["cancellation_email_screenshot.png"]
            merch_ev = ["terms_of_service.pdf", "account_activity_log.json"]
            
        elif reason == "unrecognized_merchant":
            cust_stmt = f"I do not recognize merchant name 'StoreFront #{6000 + i}' or charge reference ref_{7000 + i} on my statement."
            merch_stmt = f"Merchant name 'StoreFront #{6000 + i}' matches the parent DBA registered to the legal entity processing reference ref_{7000 + i}."
            cust_ev = ["statement_highlight.png"]
            merch_ev = ["merchant_registration_license.pdf", "fulfillment_invoice.pdf"]

        elif reason == "wrong_item_shipped":
            cust_stmt = f"I ordered a blue jacket for order #{8000 + i}, but received a completely different red shirt instead."
            merch_stmt = f"Warehouse inventory logs confirm the exact SKU ordered for #{8000 + i} was packed and dispatched."
            cust_ev = ["wrong_item_photo.png"]
            merch_ev = ["warehouse_packing_slip.pdf"]

        elif reason == "returned_item_no_refund":
            cust_stmt = f"I mailed back the return for order #{9000 + i} weeks ago via tracking, but have received no refund."
            merch_stmt = f"We inspected the returned package for #{9000 + i}, and the item was heavily used, violating return eligibility."
            cust_ev = ["return_tracking_receipt.pdf"]
            merch_ev = ["inspection_rejection_photo.png"]

        elif reason == "service_not_rendered":
            cust_stmt = f"The booking or consultation service for appointment #{1100 + i} was cancelled by the provider and never fulfilled."
            merch_stmt = f"Our staff was present at appointment #{1100 + i} at the scheduled time, but the client failed to show up."
            cust_ev = ["cancellation_notice.pdf"]
            merch_ev = ["provider_attendance_log.pdf"]

        else: # price_discrepancy
            cust_stmt = f"I was billed $120.00 for order #{1200 + i}, but the checkout promotional price shown was $90.00."
            merch_stmt = f"Promotional pricing expired prior to checkout completion for order #{1200 + i}; regular catalog pricing applies."
            cust_ev = ["checkout_screenshot.png"]
            merch_ev = ["pricing_terms_agreement.pdf"]

        case = {
            "case_id": f"case_{i:03d}",
            "dispute_reason_category": reason,
            "customer_statement": cust_stmt,
            "merchant_statement": merch_stmt,
            "evidence_customer": cust_ev,
            "evidence_merchant": merch_ev,
            "label": label
        }
        dataset.append(case)
        
    return dataset

if __name__ == "__main__":
    data = generate_synthetic_disputes(300)
    with open("disputes_dataset.json", "w") as f:
        json.dump(data, f, indent=4)
    print(f"Successfully generated {len(data)} synthetic cases across 10 reasons and saved to disputes_dataset.json")