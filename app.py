import re
import logging
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
import cohere
from rank_bm25 import BM25Okapi

COHERE_API_KEY = "eSirIQSBVNgwUhNRIv5DqK0w18ZHEYgnguNptKKV"
PORT = 8080

CONTEXT_DOCS = [
     {
        "text_chunk": "Kase Printing is a premier commercial printing company with over 30 years of experience. We provide nationwide and international offset, digital, and wide format printing. Our advanced Heidelberg Speedmaster presses offer print runs up to 16,500 sheets per hour for high-quality catalogs, brochures, and marketing materials."
    },
    {
        "text_chunk": "Our digital printing solutions excel at short-run, high-resolution projects such as postcards, flyers, and personalized print campaigns. We utilize 1200 x 1200 dpi printing technology to deliver vibrant and sharp imagery with customizable variable data options."
    },
    {
        "text_chunk": "The company is committed to G7 Master Color Certified printing ensuring consistent color accuracy through automated closed-loop color controls on the press line."
    },
    {
        "text_chunk": "Being FSC-certified distinguishes us as an environmentally conscious printer. We use sustainably sourced premium papers and non-toxic soy and vegetable-based inks, supplemented by water-based coatings and energy-efficient production processes."
    },
    {
        "text_chunk": "Direct mail and fulfillment services include postcards, brochures, catalogs, custom envelope mailers, and Every Door Direct Mail campaigns. Our in-house fulfillment capabilities allow for personalized mailings and quick turnaround logistics."
    },
    {
        "text_chunk": "Additional specialty services include foil stamping, embossing, die cutting, custom finishing, binding and kitting to complete complex print projects according to unique client specifications."
    },
    {
        "text_chunk": "Professional graphic design consultation, print project management, and marketing analytics support help clients optimize campaign effectiveness and messaging consistency across channels."
    },
    {
        "text_chunk": "Customer service is a core focus, offering expert guidance from quote to delivery ensuring every print job meets client needs and exceeds expectations."
    },
    {
        "text_chunk": "Contact our sales and service team by emailing info@kaseprinting.com or calling toll-free at 888-888-8888 for quotes, samples, or project consultation."
    }
]

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
co = cohere.Client(COHERE_API_KEY)

GREETINGS = [
    "Hello! We’re happy to help with all your printing needs at Kase Printing.",
    "Hi there! How can we assist you today?",
    "Hello and welcome to Kase Printing. What can we do for you?",
    "Hey! Let us know how we can support your print project."
]
THANKYOUS = [
    "You’re very welcome! We look forward to helping you again.",
    "Thanks for reaching out to us. We appreciate your business.",
    "Thank you! If you have more questions, we’re here to help.",
]
FALLBACKS = [
    "We’re sorry, but we didn’t quite catch that. Could you please rephrase?",
    "We’re here to help with any questions about our printing services. Could you clarify your request?",
    "That wasn’t clear to us. Can you ask another way? Our team is ready to assist.",
]
FAREWELLS = [
    "Goodbye! We appreciate your interest in Kase Printing and hope to serve you soon.",
    "Take care! Reach out anytime for printing support.",
    "Wishing you a great day from all of us at Kase Printing!"
]

def classify_intent(text):
    text = text.lower()
    if any(k in text for k in ["hello", "hi", "hey", "good morning", "good afternoon"]):
        return "greet"
    if any(k in text for k in ["thanks", "thank you", "much appreciated"]):
        return "thankyou"
    if any(k in text for k in ["contact", "phone", "email", "reach", "talk"]):
        return "contact"
    if any(k in text for k in ["bye", "goodbye", "see you"]):
        return "farewell"
    return "fallback"

def intent_response(intent):
    if intent == "greet":
        return random.choice(GREETINGS)
    if intent == "thankyou":
        return random.choice(THANKYOUS)
    if intent == "contact":
        return "You can reach us at sales@kaseprinting.com or Call At Our Direct Sales: (603) 689-1043 | General Inquiries: (603) 883-9223 We’re here for your printing projects."
    if intent == "farewell":
        return random.choice(FAREWELLS)
    if intent == "fallback":
        return random.choice(FALLBACKS)

def _tok(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())

def search_docs(query, k=3):
    corpus = [d["text_chunk"] for d in CONTEXT_DOCS]
    bm25 = BM25Okapi([_tok(d) for d in corpus])
    scores = bm25.get_scores(_tok(query))
    sorted_docs = [CONTEXT_DOCS[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)]
    return sorted_docs[:k]

@app.post("/chat")
def chat():
    data = request.get_json(force=True)
    msg = (data.get("message") or data.get("question") or "").strip()
    if not msg:
        return jsonify({"response": "Please enter a message so we can assist you."})

    intent = classify_intent(msg)
    if intent != "fallback" and intent != "contact":
        return jsonify({"response": intent_response(intent)})
    if intent == "contact":
        return jsonify({"response": intent_response("contact")})

    docs = search_docs(msg)
    context = "\n\n".join([d["text_chunk"] for d in docs])

    prompt = f"Answer as Kase Printing, speaking as the company (use 'we', 'our'), based only on the information below:\n{context}\n\nQuestion: {msg}\nAnswer:"

    try:
        response = co.chat(
            model="command-r-08-2024",  # Use your Cohere-enabled model
            message=msg,
            preamble=prompt,
            max_tokens=300,
            temperature=0.3
        )
        answer = response.text.strip()
        if not answer:
            answer = intent_response("fallback")
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        answer = ("We’re currently having trouble generating an answer. "
                  "Please try again later or contact us at sales@kaseprinting.com.")

    return jsonify({"response": answer})

@app.get("/healthz")
def healthz():
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
