import sqlite3
from sqlite3 import Error
from datetime import datetime

class DatabaseManager:

    def __init__(self, db_title):
        self.db_title = db_title
        self.conn = None

    def check_database(self):
        # checking if the database exists
        try:
            self.conn = sqlite3.connect(self.db_title, uri=True, check_same_thread=False)
            # print(sqlite3.version)

        except Error as e:
            print(e)

    def close_connection(self):
        if self.conn is not None:
            self.conn.close()

    def create_user_table(self):
        self.cur = self.conn.cursor()

        # creating user table to store user details
        user_table_sql = """CREATE TABLE IF NOT EXISTS user_table (
                            actives integer,
                            first_name text,
                            email text
                            )"""

        self.cur.execute(user_table_sql)
        self.conn.commit()
        # print('User Table successfully created')

    def create_user_feedback_table(self):
        self.cur = self.conn.cursor()

        # creating user table to store user details
        feedback_table_sql = """CREATE TABLE IF NOT EXISTS user_feedback (
                            email text,
                            feedback text,
                            timestamp text
                            )"""

        self.cur.execute(feedback_table_sql)
        self.conn.commit()
        # print('Feedback Table successfully created')


    def create_chat_history_table(self):
        self.cur = self.conn.cursor()

        chat_history_sql = """CREATE TABLE IF NOT EXISTS chat_history (
                                user_text text,
                                bot_answer text,
                                topic text,
                                emotion_detected text,
                                tag text
                                )"""

        self.cur.execute(chat_history_sql)
        self.conn.commit()
        # print('User Chat History table successfully created')


    def insert_user_table(self, first_name, email):
        self.cur = self.conn.cursor()

        self.first_name = first_name
        self.email = email

        # check if table is empty
        check_query = "SELECT count(*) FROM (select 0 from user_table limit 1)"
        self.cur.execute(check_query)
        data = self.cur.fetchall()

        if data[0][0] == 0:
            # set actives as 1 indicating first user activity
            self.actives = 0
        else:
            # else filter last active number from table and increment by 1
            filter_query = "SELECT *, max(actives) FROM user_table"
            self.cur.execute(filter_query)
            max_num_of_actives = self.cur.fetchall()
            self.actives = max_num_of_actives[0][0] + 1

        query = "INSERT OR IGNORE INTO user_table values (?, ?, ?)"

        self.cur.execute(query,(self.actives, self.first_name, self.email))
        self.conn.commit()

        # print('Record successfully inserted in User Table')


    def insert_feedback_table(self, email, feedback):
        self.cur = self.conn.cursor()
        self.feedback = feedback
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.email = email

        insert_query = "INSERT OR IGNORE INTO user_feedback values (?, ?, ?)"
        self.cur.execute(insert_query, (self.email, self.feedback, self.timestamp))
        self.conn.commit()
        # print('Record successfully added in User Feedback table')

    def insert_chat_table(self, user_text, bot_text, topic, emotion_detected, tag):
        self.cur = self.conn.cursor()

        self.user_text = user_text
        self.bot_text = bot_text

        self.topic = topic
        self.emotion_detected = emotion_detected
        self.tag = tag

        insert_query = "INSERT OR IGNORE INTO chat_history values (?, ?, ?, ?, ?)"
        self.cur.execute(insert_query, (self.user_text, self.bot_text, self.topic, self.emotion_detected, self.tag))
        self.conn.commit()

        # print('Record successfully added in Chat History Table')

    def fetch_user_name(self):
        self.cur = self.conn.cursor()
        self.cur.execute('SELECT * FROM user_table ORDER BY rowid DESC LIMIT 1')
        row = self.cur.fetchall()
        user_name = row[0][1]
        return user_name

    def fetch_user_email(self):
        self.cur = self.conn.cursor()
        self.cur.execute('SELECT * FROM user_table ORDER BY rowid DESC LIMIT 1')
        row = self.cur.fetchall()
        user_email = row[0][2]
        return user_email

    def fetch_last_emotion(self):
        self.cur = self.conn.cursor()
        self.cur.execute('SELECT * FROM chat_history ORDER BY rowid DESC LIMIT 1')
        row = self.cur.fetchall()
        last_emotion = row[0][3]
        return last_emotion

    def fetch_previous_tag(self):
        self.cur = self.conn.cursor()
        # check if table is empty
        check_query = "SELECT count(*) FROM (select 0 from chat_history limit 1)"
        self.cur.execute(check_query)
        data = self.cur.fetchall()

        if data[0][0] == 0:
            previous_tag = 'empty'
        elif data[0][0] == 1:
            self.cur.execute('SELECT * FROM chat_history ORDER BY rowid DESC LIMIT 2')
            row = self.cur.fetchall()
            previous_tag = row[0][4]
        else:
            self.cur.execute('SELECT * FROM chat_history ORDER BY rowid DESC LIMIT 2')
            row = self.cur.fetchall()
            previous_tag = row[1][4]

        return previous_tag

    def fetch_active(self):
        self.cur = self.conn.cursor()

        self.cur.execute('SELECT * FROM user_table ORDER BY actives DESC LIMIT 1')
        row = self.cur.fetchall()
        active = row[0][0]

        return active

    def fetch_question_answer(self):
        self.cur = self.conn.cursor()

        self.cur.execute('SELECT user_text, bot_answer FROM chat_history')
        row = self.cur.fetchall()

        return row

def main():
    db_manager = DatabaseManager('EmpathBot.db')
    db_manager.check_database()
    db_manager.create_user_table()
    db_manager.create_user_feedback_table()
    db_manager.create_chat_history_table()
    # n = db_manager.fetch_user_name()
    # e = db_manager.fetch_last_emotion()
    # print(e)
    # p = db_manager.fetch_previous_tag()
    # print(p)
    l = db_manager.fetch_question_answer()
    print(l)

    db_manager.close_connection()


if __name__ == '__main__':
    main()
