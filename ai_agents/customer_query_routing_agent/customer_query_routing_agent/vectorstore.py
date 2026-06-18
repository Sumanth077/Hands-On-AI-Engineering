"""
VectorAI DB client, collection setup, and seeding.

Four collections forming the unified context layer:
  product_faq       - clean Q&A pairs from the support knowledge base
  product_docs      - policy and documentation chunks (return policy, billing T&Cs, etc.)
  resolved_tickets  - historical support ticket threads with resolutions
  resolved_queries  - persistent agent memory; grows as queries are resolved

All four are searched on every incoming query. The router combines results
and labels them by source before passing them to the LLM.
"""

from __future__ import annotations

from actian_vectorai import (
    Distance,
    PointStruct,
    VectorAIClient,
    VectorParams,
)

from customer_query_routing_agent.config import (
    DOCS_COLLECTION,
    EMBEDDING_DIM,
    FAQ_COLLECTION,
    MEMORY_COLLECTION,
    TICKETS_COLLECTION,
    VECTORAI_ACCESS_TOKEN,
    VECTORAI_URL,
)


# ---------------------------------------------------------------------------
# Source 1: FAQ knowledge base
# ---------------------------------------------------------------------------

FAQ_SEED: list[dict] = [
    # Returns & Refunds
    {
        "question": "How do I return a product?",
        "answer": (
            "To return a product, go to Orders in your account, select the item, "
            "and click Return Item. Print the prepaid label and drop it at any courier location. "
            "Refunds are processed within 5-7 business days after we receive the item."
        ),
        "department": "Returns & Refunds",
    },
    {
        "question": "What is your return policy?",
        "answer": (
            "We accept returns within 30 days of delivery for most items in their original condition. "
            "Electronics must be unopened. Perishables and personalised items are non-returnable. "
            "If the item arrived damaged, we cover return shipping."
        ),
        "department": "Returns & Refunds",
    },
    {
        "question": "My item arrived damaged, what do I do?",
        "answer": (
            "Take a photo of the damage and go to Orders > Report Problem within 48 hours of delivery. "
            "We will send a replacement or issue a full refund, your choice, at no additional cost."
        ),
        "department": "Returns & Refunds",
    },
    {
        "question": "How long does a refund take?",
        "answer": (
            "Refunds appear within 5-7 business days after we confirm receipt of the returned item. "
            "Original payment method is always used. Your bank may take an extra 1-2 days to post the credit."
        ),
        "department": "Returns & Refunds",
    },
    {
        "question": "Can I exchange an item instead of returning it?",
        "answer": (
            "Yes. During the return flow, select Exchange and choose the replacement size or colour. "
            "Exchanges ship free and the original must be returned within 14 days."
        ),
        "department": "Returns & Refunds",
    },
    # Billing & Payments
    {
        "question": "Why was I charged twice?",
        "answer": (
            "A duplicate charge is usually a temporary authorisation hold, not a permanent charge. "
            "It should disappear within 3-5 business days. If it has not cleared, contact us with "
            "the transaction dates and amounts so we can raise a dispute."
        ),
        "department": "Billing & Payments",
    },
    {
        "question": "What payment methods do you accept?",
        "answer": (
            "We accept Visa, Mastercard, American Express, PayPal, Apple Pay, and Google Pay. "
            "Gift cards are also accepted at checkout."
        ),
        "department": "Billing & Payments",
    },
    {
        "question": "My payment was declined, what should I do?",
        "answer": (
            "Payment declines are usually caused by incorrect billing address, expired card, or "
            "your bank flagging an unusual purchase. Double-check your details, then try again. "
            "If the problem persists, use a different payment method or contact your bank."
        ),
        "department": "Billing & Payments",
    },
    {
        "question": "Can I get an invoice for my order?",
        "answer": (
            "Go to Orders, open the relevant order, and click Download Invoice. "
            "A PDF invoice with a VAT breakdown is generated instantly."
        ),
        "department": "Billing & Payments",
    },
    {
        "question": "How do I update my payment method?",
        "answer": (
            "Go to Account > Payment Methods to add, remove, or set a default card. "
            "For a subscription, update before your next billing date to avoid interruption."
        ),
        "department": "Billing & Payments",
    },
    # Technical Support
    {
        "question": "I cannot log in to my account",
        "answer": (
            "Click Forgot Password on the login page and enter your email address. "
            "You will receive a reset link within a few minutes. Check your spam folder "
            "if it does not arrive. If you no longer have access to that email, contact support."
        ),
        "department": "Technical Support",
    },
    {
        "question": "The app keeps crashing, how do I fix it?",
        "answer": (
            "Force-close the app and reopen it. If that does not help, clear the app cache "
            "in your phone settings. Make sure you are on the latest version. As a last resort, "
            "uninstall and reinstall. Your account data is stored in the cloud."
        ),
        "department": "Technical Support",
    },
    {
        "question": "I am not receiving email notifications",
        "answer": (
            "Check spam and promotions folders first. If emails are landing there, mark them "
            "as Not Spam. Also verify that notifications are enabled under Account > Notifications."
        ),
        "department": "Technical Support",
    },
    {
        "question": "How do I reset my password?",
        "answer": (
            "On the login screen, click Forgot Password and enter the email on your account. "
            "We send a secure reset link valid for 30 minutes. The link can only be used once."
        ),
        "department": "Technical Support",
    },
    {
        "question": "The website is loading slowly or not working",
        "answer": (
            "Try clearing your browser cache and cookies, then reload the page. "
            "If you are using an ad blocker or VPN, try disabling it temporarily."
        ),
        "department": "Technical Support",
    },
    # Order Tracking
    {
        "question": "Where is my order?",
        "answer": (
            "Go to Orders and click Track Shipment next to your order. "
            "A tracking link was also emailed to you when your order shipped. "
            "Standard delivery takes 3-5 business days."
        ),
        "department": "Order Tracking",
    },
    {
        "question": "My tracking number is not working",
        "answer": (
            "Tracking numbers can take up to 24 hours to activate after your order ships. "
            "If it still does not work after 24 hours, contact us if there is no update after 3 business days."
        ),
        "department": "Order Tracking",
    },
    {
        "question": "Can I change my delivery address after ordering?",
        "answer": (
            "Address changes are possible within 1 hour of placing the order. "
            "Go to Orders > Manage Order > Edit Address. After 1 hour the order goes to the warehouse "
            "and we can no longer redirect it."
        ),
        "department": "Order Tracking",
    },
    {
        "question": "My order says delivered but I did not receive it",
        "answer": (
            "Check around your property including any safe spots noted in your delivery preferences. "
            "Ask neighbours in case it was left with them. If still not found, "
            "report it within 72 hours via Orders > Report Missing Delivery."
        ),
        "department": "Order Tracking",
    },
    {
        "question": "How do I cancel an order?",
        "answer": (
            "Orders can be cancelled from Orders > Cancel Order within 1 hour of purchase. "
            "After that, the order enters fulfilment and cannot be cancelled. "
            "You would need to return it once delivered."
        ),
        "department": "Order Tracking",
    },
    # General Inquiry
    {
        "question": "How do I contact customer support?",
        "answer": (
            "Live chat on our website (9am-9pm Mon-Sun), "
            "email at support@example.com (response within 24 hours), "
            "or phone at 1-800-555-0100 (9am-6pm Mon-Fri)."
        ),
        "department": "General Inquiry",
    },
    {
        "question": "Do you offer a loyalty or rewards programme?",
        "answer": (
            "Yes. Every purchase earns points that convert to store credit. "
            "Sign up or check your balance under Account > Rewards. Points expire 12 months after being earned."
        ),
        "department": "General Inquiry",
    },
    {
        "question": "Do you ship internationally?",
        "answer": (
            "We ship to over 40 countries. Delivery times and fees vary by destination and are shown at checkout. "
            "Import duties and taxes are the responsibility of the recipient."
        ),
        "department": "General Inquiry",
    },
    {
        "question": "How do I apply a discount code?",
        "answer": (
            "Enter your code in the Promo Code field at checkout before completing payment. "
            "One code per order. Codes cannot be applied after an order is placed."
        ),
        "department": "General Inquiry",
    },
    {
        "question": "Is my personal data safe with you?",
        "answer": (
            "Yes. We follow GDPR and CCPA requirements. Your data is encrypted in transit and at rest. "
            "We never sell your data to third parties. You can request a copy or deletion under Account > Privacy."
        ),
        "department": "General Inquiry",
    },
]


# ---------------------------------------------------------------------------
# Source 2: Product documentation chunks
# ---------------------------------------------------------------------------

PRODUCT_DOCS_SEED: list[dict] = [
    {
        "title": "Return and Refund Policy",
        "content": (
            "Our return policy covers most items purchased on the platform within 30 days of the "
            "confirmed delivery date. Items must be in their original condition with all original "
            "packaging and tags attached. Electronics may only be returned if the box is unopened "
            "and the factory seal is intact. Items marked as Final Sale, perishables, digital "
            "downloads, personalised or custom-made products, and hazardous materials are excluded "
            "from our standard return policy. "
            "To initiate a return, customers must log in to their account, navigate to Orders, select "
            "the item, and choose Return Item. The system generates a prepaid return label for eligible "
            "orders. Customers must drop the parcel at an approved courier location within 7 days of "
            "generating the label, or the return request will expire. "
            "Refunds are issued to the original payment method and typically appear within 5-7 business "
            "days after our warehouse confirms receipt and condition of the returned item. Credit card "
            "refunds may require an additional 1-3 business days to reflect on the statement, depending "
            "on the issuing bank."
        ),
        "department": "Returns & Refunds",
        "doc_type": "policy",
    },
    {
        "title": "Shipping and Delivery Policy",
        "content": (
            "Standard domestic shipping takes 3-5 business days from the date the order is dispatched. "
            "Express shipping (1-2 business days) is available at checkout for an additional fee. "
            "Same-day delivery is offered in selected metro areas for orders placed before 11am local time. "
            "Orders are processed Monday through Friday, excluding public holidays. Orders placed after "
            "3pm on a business day are processed the following business day. "
            "Once dispatched, customers receive a shipping confirmation email containing a tracking number "
            "and a direct link to the carrier's tracking page. Tracking information may take up to "
            "24 hours to become active on the carrier's system after the confirmation email is sent. "
            "International shipping is available to 44 countries. Delivery times for international "
            "orders vary between 7-21 business days depending on the destination and chosen service. "
            "All applicable import duties, customs taxes, and brokerage fees are the sole responsibility "
            "of the recipient and are not included in the order total at checkout. "
            "We are not responsible for delays caused by customs processing, weather events, or "
            "carrier disruptions beyond our control."
        ),
        "department": "Order Tracking",
        "doc_type": "policy",
    },
    {
        "title": "Billing and Subscription Terms",
        "content": (
            "Subscriptions are billed on a recurring cycle - monthly or annual - on the same calendar "
            "date as the original sign-up. Annual plans are billed in full at the start of each billing "
            "cycle and offer a 20% discount versus the monthly rate. "
            "Payment is collected automatically using the default payment method on file. Customers "
            "should ensure their card details are current before the billing date to avoid service "
            "interruption. If a payment fails, we retry up to three times over five business days. "
            "After three failed attempts, access to premium features is suspended until the outstanding "
            "balance is settled. "
            "Customers may cancel their subscription at any time from Account > Subscription > Cancel. "
            "Cancellations take effect at the end of the current billing period. No partial refunds are "
            "issued for unused days in a billing period except where required by local consumer law. "
            "Downgrades from annual to monthly plans take effect at the next renewal date. "
            "Promotional rates and introductory pricing apply only for the period stated at sign-up "
            "and revert to the standard rate at renewal unless a new promotion is active."
        ),
        "department": "Billing & Payments",
        "doc_type": "policy",
    },
    {
        "title": "Warranty and Product Guarantee",
        "content": (
            "All products sold through our platform are covered by the manufacturer's standard warranty. "
            "Warranty periods vary by product category: electronics carry a minimum 12-month warranty, "
            "appliances 24 months, and clothing or accessories 90 days for manufacturing defects. "
            "Our platform warranty covers defects in materials and workmanship under normal use conditions. "
            "It does not cover damage resulting from accidental drops, liquid contact, unauthorised "
            "modifications, misuse, or normal wear and tear. "
            "To make a warranty claim, customers must contact support within the warranty period and "
            "provide proof of purchase, a description of the defect, and photographs where applicable. "
            "Our team will assess the claim within 3 business days and may request the item be returned "
            "for inspection. Approved warranty claims are resolved by repair, replacement, or store credit "
            "at our discretion. "
            "Third-party warranties or extended protection plans purchased at checkout are managed "
            "directly by the insurer and may have different claim procedures."
        ),
        "department": "Returns & Refunds",
        "doc_type": "policy",
    },
    {
        "title": "Account Security and Two-Factor Authentication",
        "content": (
            "We strongly recommend enabling two-factor authentication (2FA) on all accounts. "
            "2FA can be activated from Account > Security and requires a valid mobile number or "
            "authenticator app. Once enabled, a verification code is required at every login from "
            "an unrecognised device. "
            "Passwords must be a minimum of 8 characters and include at least one uppercase letter, "
            "one number, and one special character. Passwords are hashed and salted using industry-standard "
            "algorithms and are never stored in plaintext. We will never ask for your password via "
            "email, chat, or phone. "
            "If you suspect your account has been accessed without your permission, immediately change "
            "your password and contact support. We will place a security hold on the account while "
            "investigating. During a security hold, no purchases, address changes, or payment method "
            "changes can be made until identity is verified. "
            "For forgotten passwords, use the Forgot Password link on the login screen. Reset links "
            "expire after 30 minutes and can only be used once."
        ),
        "department": "Technical Support",
        "doc_type": "policy",
    },
    {
        "title": "Privacy Policy Summary",
        "content": (
            "We collect personal data including name, email address, shipping address, payment details, "
            "and browsing behaviour solely to fulfil orders and improve our services. We do not sell, "
            "rent, or trade your personal data to third parties for marketing purposes. "
            "Data is stored on encrypted servers hosted in ISO 27001-certified data centres. "
            "We comply with GDPR for EU residents and CCPA for California residents. Under these "
            "regulations, you have the right to access, correct, or delete your personal data at any time. "
            "Submit a data request from Account > Privacy > Manage My Data. Requests are completed "
            "within 30 days. "
            "We use cookies and similar tracking technologies to personalise your experience and measure "
            "the effectiveness of our marketing. You can manage cookie preferences in the banner that "
            "appears on your first visit or from Account > Privacy > Cookie Settings. "
            "We share data with payment processors (for transaction completion), logistics partners "
            "(for delivery), and analytics providers under strict data processing agreements."
        ),
        "department": "General Inquiry",
        "doc_type": "policy",
    },
    {
        "title": "Loyalty and Rewards Programme Terms",
        "content": (
            "The Rewards Programme is free to join for all registered customers. Points are earned "
            "at a rate of 1 point per dollar spent on eligible purchases, excluding taxes, shipping "
            "fees, gift card purchases, and items bought with existing store credit. "
            "Points can be redeemed as store credit at checkout at a rate of 100 points = $1.00. "
            "A minimum of 500 points is required to redeem. Partial redemptions are allowed in "
            "increments of 100 points. "
            "Points expire 12 months after the date they were earned if the account has had no "
            "qualifying purchase activity in that period. Points earned on orders that are later "
            "returned or refunded will be deducted from the account balance. "
            "Bonus point events, referral bonuses, and birthday rewards are issued at our discretion "
            "and may be subject to additional terms communicated at the time of the offer. "
            "We reserve the right to modify point earn and redemption rates, terminate the programme, "
            "or cancel an account's points if fraudulent activity is detected."
        ),
        "department": "General Inquiry",
        "doc_type": "policy",
    },
    {
        "title": "Disputes and Chargebacks",
        "content": (
            "If you believe a charge is incorrect, please contact our billing team before initiating "
            "a chargeback with your bank. Most billing disputes can be resolved within 2-3 business days "
            "by our team, significantly faster than the bank's chargeback process which typically "
            "takes 30-90 days. "
            "To dispute a charge, go to Orders > select the relevant order > Report Billing Issue, "
            "and provide the transaction date, amount, and reason for the dispute. Our team will "
            "review bank statements and order records and respond within 2 business days. "
            "If a chargeback is filed without contacting us first, access to the account is "
            "temporarily suspended pending investigation. If the chargeback is found to be invalid, "
            "the account may be permanently closed and any outstanding balance referred to collections. "
            "Legitimate billing errors are refunded promptly and are never contested."
        ),
        "department": "Billing & Payments",
        "doc_type": "policy",
    },
]


# ---------------------------------------------------------------------------
# Source 3: Resolved support ticket threads
# ---------------------------------------------------------------------------

RESOLVED_TICKETS_SEED: list[dict] = [
    {
        "ticket_id": "TKT-8821",
        "summary": "Customer received wrong size and wants an exchange",
        "thread": (
            "Customer: I ordered a medium hoodie but received a large. Order #47821. "
            "Can I get the right size?\n\n"
            "Agent (Sarah): Hi! I can see order #47821 - a medium hoodie shipped on March 3rd. "
            "I am sorry we sent the wrong size. I have just raised an exchange request for you. "
            "A prepaid return label has been emailed to you. Once you drop off the original item "
            "at any UPS location, we will immediately dispatch the medium. You do not need to "
            "wait for us to receive it first.\n\n"
            "Customer: That was fast, thank you!\n\n"
            "Resolution: Exchange raised for correct size, prepaid label issued, replacement "
            "dispatched within 24 hours of drop-off. Ticket closed after customer confirmed receipt."
        ),
        "department": "Returns & Refunds",
        "resolution_type": "exchange",
    },
    {
        "ticket_id": "TKT-9103",
        "summary": "Customer charged twice for the same order, requesting refund of duplicate",
        "thread": (
            "Customer: I was charged $89.99 twice on March 7th for order #51002. "
            "My bank statement shows two identical charges.\n\n"
            "Agent (James): I can see order #51002 for $89.99. I am checking the payment logs now. "
            "I can confirm there was a system error during checkout that created a duplicate "
            "authorisation. The second charge of $89.99 was a permanent capture, not just a hold. "
            "I have raised a refund for the duplicate amount now. You should see it return to your "
            "card within 5-7 business days. I am also adding a $10 credit to your account as "
            "an apology for the inconvenience.\n\n"
            "Customer: Thank you for sorting it out quickly.\n\n"
            "Resolution: Duplicate charge confirmed via payment logs. Refund of $89.99 issued. "
            "Courtesy $10 account credit applied. Root cause escalated to payments engineering."
        ),
        "department": "Billing & Payments",
        "resolution_type": "refund",
    },
    {
        "ticket_id": "TKT-9354",
        "summary": "App crashing on iOS 18 after recent update, customer unable to access orders",
        "thread": (
            "Customer: Your app has been crashing every time I open it since the update yesterday. "
            "I am on an iPhone 15, iOS 18.2. I cannot see my orders at all.\n\n"
            "Agent (Priya): I am sorry about this. We are aware of a compatibility issue with iOS 18.2 "
            "and our latest app update (v4.1.2) that was released yesterday. Our engineering team "
            "is working on a patch. In the meantime, here is a workaround: go to Settings > "
            "General > iPhone Storage, find our app, and tap Offload App. Then reinstall from "
            "the App Store - this installs the previous stable version (v4.1.0) while the patch "
            "is in review. You will not lose any data.\n\n"
            "Customer: The offload and reinstall worked. App is back to normal.\n\n"
            "Resolution: Known iOS 18.2 crash bug in v4.1.2. Workaround: offload and reinstall. "
            "Patch v4.1.3 submitted to App Store review. Bug logged in engineering tracker."
        ),
        "department": "Technical Support",
        "resolution_type": "workaround_provided",
    },
    {
        "ticket_id": "TKT-9561",
        "summary": "Package marked as delivered but customer never received it",
        "thread": (
            "Customer: My order #54309 shows delivered yesterday but I never got it. "
            "I checked with my neighbours and nobody has it either.\n\n"
            "Agent (Marcus): I can see order #54309 - the carrier (FedEx) marked it as delivered "
            "at 2:14pm yesterday with a note saying it was left at the front door. "
            "I have opened a trace investigation with FedEx. These take 2-3 business days. "
            "Rather than making you wait, I will go ahead and send a replacement order now. "
            "If FedEx recovers the original package you can simply refuse delivery. "
            "The replacement is being expedited and should arrive within 2 business days.\n\n"
            "Customer: That is great, thank you for not making me wait for the investigation.\n\n"
            "Resolution: Missing delivery confirmed. Replacement dispatched immediately on expedited "
            "shipping. FedEx trace investigation opened. Original package not recovered. "
            "Replacement delivered and confirmed by customer."
        ),
        "department": "Order Tracking",
        "resolution_type": "replacement_sent",
    },
    {
        "ticket_id": "TKT-9782",
        "summary": "Customer wants to cancel annual subscription and get a prorated refund",
        "thread": (
            "Customer: I signed up for the annual plan 3 months ago but I am not using the service "
            "anymore. Can I cancel and get a refund for the remaining 9 months?\n\n"
            "Agent (Lisa): I understand. Looking at your account, your annual plan was charged on "
            "December 10th for $119.99. Our standard policy does not cover prorated refunds for "
            "annual plans after the first 30 days, as stated in our billing terms. However, "
            "I can see you have only logged in twice since signing up, so I am going to make an "
            "exception and refund 6 months of the remaining value - $59.99 - as a goodwill gesture. "
            "The subscription will be cancelled at the end of today. Would that work for you?\n\n"
            "Customer: Yes, I appreciate that. Thank you.\n\n"
            "Resolution: Partial goodwill refund of $59.99 issued (6 of 9 remaining months). "
            "Subscription cancelled immediately. Exception approved by supervisor. "
            "Note: customer was inactive - low churn risk to re-acquire."
        ),
        "department": "Billing & Payments",
        "resolution_type": "partial_refund_goodwill",
    },
    {
        "ticket_id": "TKT-9901",
        "summary": "Promo code not applying at checkout, customer getting error",
        "thread": (
            "Customer: I have a 20% off code WELCOME20 but every time I enter it at checkout "
            "it says the code is invalid.\n\n"
            "Agent (Tom): Let me check that code for you. I can see WELCOME20 is valid but it has "
            "a restriction: it only applies to your first order and I can see you placed an order "
            "in October last year, so the code has already been used on your account. "
            "I know that is frustrating if you did not intentionally use it. "
            "Since it looks like the October order was a small gift card purchase, I am going to "
            "apply a one-time 15% discount to your current cart manually. Give me a moment and "
            "I will send you a single-use code.\n\n"
            "Customer: AGENT15OFF worked. Placed the order. Thank you!\n\n"
            "Resolution: Original code was used against policy intent (gift card purchase). "
            "One-time 15% agent code AGENT15OFF issued as goodwill. Order placed successfully."
        ),
        "department": "Billing & Payments",
        "resolution_type": "manual_discount_applied",
    },
    {
        "ticket_id": "TKT-10044",
        "summary": "Customer cannot reset password, reset link says it has expired",
        "thread": (
            "Customer: I requested a password reset but every link I click says it has already expired. "
            "I have tried three times.\n\n"
            "Agent (Nina): Password reset links expire after 30 minutes and are single-use. "
            "If you are copying the link from an email preview pane or a mobile client that "
            "pre-fetches links, the link gets consumed before you click it. "
            "Could you try this: open the email on a desktop browser, right-click the link, "
            "copy the URL, and paste it into a new tab. Do not click it directly. "
            "If that does not work, I can send a reset link directly to your screen via our "
            "support portal - it bypasses the email client issue entirely.\n\n"
            "Customer: The copy-paste method worked! I am in now.\n\n"
            "Resolution: Link pre-fetch by mobile email client was consuming the single-use token. "
            "Workaround: right-click copy URL and paste into browser. Common iOS Mail issue. "
            "Added to known issues doc."
        ),
        "department": "Technical Support",
        "resolution_type": "workaround_provided",
    },
    {
        "ticket_id": "TKT-10187",
        "summary": "International customer asking about import duties charged at delivery",
        "thread": (
            "Customer: I ordered from Australia and was charged $43 in import duties at delivery "
            "that I was not told about at checkout. Why was I not informed?\n\n"
            "Agent (Chen): I am sorry this was not clear at checkout. For international orders, "
            "our checkout displays the product price and shipping fee but it cannot calculate "
            "customs and import duties in advance as these are determined by your local customs "
            "authority based on the declared value and product category. Our shipping policy "
            "and checkout page do note that duties may apply, but I understand it is easy to miss. "
            "We cannot refund import duties as they are paid to a government authority, not to us. "
            "What I can do is apply a $20 store credit to your account as a goodwill gesture, "
            "and I will also flag this feedback to improve how we communicate duty obligations "
            "to international customers at checkout.\n\n"
            "Customer: I appreciate the credit and the honest explanation.\n\n"
            "Resolution: Import duty education provided. $20 goodwill credit applied. "
            "Feedback passed to product team to improve international checkout duty disclosure."
        ),
        "department": "General Inquiry",
        "resolution_type": "education_and_goodwill_credit",
    },
]


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def get_client() -> VectorAIClient:
    """Return a connected VectorAI DB client."""
    kwargs: dict = {"url": VECTORAI_URL}
    if VECTORAI_ACCESS_TOKEN:
        kwargs["access_token"] = VECTORAI_ACCESS_TOKEN
    return VectorAIClient(**kwargs)


# ---------------------------------------------------------------------------
# Collection setup
# ---------------------------------------------------------------------------

COLLECTIONS = [FAQ_COLLECTION, DOCS_COLLECTION, TICKETS_COLLECTION, MEMORY_COLLECTION]


def setup_collections(client: VectorAIClient) -> None:
    """Create all four collections if they do not already exist."""
    vector_cfg = VectorParams(size=EMBEDDING_DIM, distance=Distance.Cosine)
    for name in COLLECTIONS:
        if not client.collections.exists(name):
            client.collections.create(name, vectors_config=vector_cfg)
            print(f"Created collection: {name}")


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def _upsert_texts(
    client: VectorAIClient,
    collection: str,
    records: list[dict],
    text_key: str,
    embedder,
    start_id: int = 1,
) -> None:
    """Embed `text_key` from each record and upsert into the collection."""
    texts = [r[text_key] for r in records]
    vectors = embedder.embed_batch(texts)
    points = [
        PointStruct(id=start_id + i, vector=vectors[i], payload=records[i])
        for i in range(len(records))
    ]
    client.points.upsert(collection, points)


def seed_all(client: VectorAIClient, embedder) -> None:
    """
    Seed all three static knowledge sources on first run.
    Each collection is only seeded if it is currently empty (idempotent).
    """
    if client.points.count(FAQ_COLLECTION) == 0:
        print(f"Seeding {len(FAQ_SEED)} FAQ entries...")
        _upsert_texts(client, FAQ_COLLECTION, FAQ_SEED, "question", embedder)
        print("FAQ knowledge base ready.")

    if client.points.count(DOCS_COLLECTION) == 0:
        print(f"Seeding {len(PRODUCT_DOCS_SEED)} product documentation chunks...")
        _upsert_texts(client, DOCS_COLLECTION, PRODUCT_DOCS_SEED, "content", embedder)
        print("Product docs collection ready.")

    if client.points.count(TICKETS_COLLECTION) == 0:
        print(f"Seeding {len(RESOLVED_TICKETS_SEED)} resolved ticket threads...")
        _upsert_texts(client, TICKETS_COLLECTION, RESOLVED_TICKETS_SEED, "summary", embedder)
        print("Resolved tickets collection ready.")


# ---------------------------------------------------------------------------
# Search helpers - one per source
# ---------------------------------------------------------------------------

def search_faq(client: VectorAIClient, vector: list[float], top_k: int = 3) -> list:
    return client.points.search(FAQ_COLLECTION, vector=vector, limit=top_k)


def search_docs(client: VectorAIClient, vector: list[float], top_k: int = 2) -> list:
    return client.points.search(DOCS_COLLECTION, vector=vector, limit=top_k)


def search_tickets(client: VectorAIClient, vector: list[float], top_k: int = 2) -> list:
    return client.points.search(TICKETS_COLLECTION, vector=vector, limit=top_k)


def search_memory(client: VectorAIClient, vector: list[float], top_k: int = 2) -> list:
    if client.points.count(MEMORY_COLLECTION) == 0:
        return []
    return client.points.search(MEMORY_COLLECTION, vector=vector, limit=top_k)


# ---------------------------------------------------------------------------
# Memory write
# ---------------------------------------------------------------------------

def store_resolved_query(
    client: VectorAIClient,
    query: str,
    query_vector: list[float],
    resolution: str,
    department: str,
) -> None:
    """
    Persist a resolved query into VectorAI DB memory.
    This collection grows over time and is searched alongside the static sources.
    """
    import time
    point_id = int(time.time() * 1000) % (2**31)
    client.points.upsert(MEMORY_COLLECTION, [
        PointStruct(
            id=point_id,
            vector=query_vector,
            payload={"query": query, "resolution": resolution, "department": department},
        )
    ])


def get_collection_counts(client: VectorAIClient) -> dict[str, int]:
    """Return document counts for all collections."""
    return {name: client.points.count(name) for name in COLLECTIONS}
