from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)


feedback_data = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stressPrediction', methods=['POST'])
def submit_feedback():
    first_name = request.form['first_name']
    email = request.form['email']
    message = request.form['message']
    
   
    feedback_data.append({
        'first_name': first_name,
        'email': email,
        'message': message
    })
    
    return redirect(url_for('thank_you'))

@app.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')

if __name__ == '__main__':
    app.run(debug=True , host='0.0.0.0')

