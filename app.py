from flask import Flask, render_template, request, jsonify
from chat import get_final_response, ChatBot
from flask_cors import CORS


from database_handler import DatabaseManager

app = Flask(__name__)
CORS(app)

db_manager = DatabaseManager('EmpathBot.db')
db_manager.check_database()

@app.get("/")
def index_get():
    return render_template("index.html")

@app.post("/get_details")
def get_details():
    user_name = request.form['userName']
    user_email = request.form['userEmail']
    db_manager.insert_user_table(user_name, user_email)
    return '', 204

@app.post("/predict")
def predict():
    user_text = request.get_json().get("message")
    response = get_final_response(user_text)

    message = {"answer": response}
    return jsonify(message)

@app.post("/user_feedback")
def user_feedback():
    if request.form['great-button'] == "great":
        db_manager.insert_feedback_table(db_manager.fetch_user_email(), "good")
    elif request.form['bad-button'] == "bad":
        db_manager.insert_feedback_table(db_manager.fetch_user_email(), "bad")
    return '', 204

if __name__ == "__main__":
    app.run()


# closing connection
db_manager.close_connection()