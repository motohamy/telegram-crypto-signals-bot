from flask import Flask, request, jsonify
import logging
import json
import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("webhook_receiver.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WebhookReceiver")

# Create Flask app
app = Flask(__name__)

# Create directory for storing webhook data
os.makedirs("webhook_data", exist_ok=True)

@app.route('/')
def home():
    return "Webhook Receiver is running!"

@app.route('/webhook/btc', methods=['POST'])
def webhook_btc():
    """Endpoint for BTC webhook"""
    data = request.json
    
    logger.info(f"Received BTC webhook: {data}")
    
    # Save to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"webhook_data/btc_{timestamp}.json", "w") as f:
        json.dump(data, f, indent=2)
    
    return jsonify({"status": "success", "message": "BTC webhook received"})

@app.route('/webhook/sol', methods=['POST'])
def webhook_sol():
    """Endpoint for SOL webhook"""
    data = request.json
    
    logger.info(f"Received SOL webhook: {data}")
    
    # Save to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"webhook_data/sol_{timestamp}.json", "w") as f:
        json.dump(data, f, indent=2)
    
    return jsonify({"status": "success", "message": "SOL webhook received"})

@app.route('/webhook/default', methods=['POST'])
def webhook_default():
    """Default endpoint for other cryptocurrencies"""
    data = request.json
    
    logger.info(f"Received default webhook: {data}")
    
    # Extract ticker if available
    ticker = "unknown"
    if data and 'text' in data:
        text_data = data['text']
        try:
            # Try to parse JSON inside the text
            import re
            json_match = re.search(r'{.*}', text_data)
            if json_match:
                parsed = json.loads(json_match.group(0))
                if 'ticker' in parsed:
                    ticker = parsed['ticker']
        except:
            pass
    
    # Save to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"webhook_data/{ticker}_{timestamp}.json", "w") as f:
        json.dump(data, f, indent=2)
    
    return jsonify({"status": "success", "message": "Default webhook received"})

def parse_webhook_data(data):
    """Parse the webhook data to extract trading information"""
    if not data or 'text' not in data:
        return None
    
    text = data['text']
    result = {
        'raw_text': text
    }
    
    # Extract action (BUY, SELL, EXIT, HOLD)
    if text.startswith('BUY '):
        result['action'] = 'BUY'
    elif text.startswith('SELL '):
        result['action'] = 'SELL'
    elif text.startswith('EXIT '):
        result['action'] = 'EXIT'
    elif text.startswith('HOLD '):
        result['action'] = 'HOLD'
    
    # Extract JSON data
    import re
    json_match = re.search(r'{.*}', text)
    if json_match:
        try:
            json_data = json.loads(json_match.group(0))
            result.update(json_data)
        except json.JSONDecodeError:
            pass
    
    return result

if __name__ == '__main__':
    print("Starting Webhook Receiver on http://localhost:5000")
    print("BTC Endpoint: http://localhost:5000/webhook/btc")
    print("SOL Endpoint: http://localhost:5000/webhook/sol")
    print("Default Endpoint: http://localhost:5000/webhook/default")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
