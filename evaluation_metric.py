import json
import nltk
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu

from data_preprocessing import text_cleaning
from database_handler import DatabaseManager
from text_similarity import cosine_similarity
# nltk.download('omw-1.4')

# return question with highest similarity to user text
def question_similarity(user_text, questions):
    max = float('-inf')

    for each in questions:
        sim = cosine_similarity(user_text, each)
        if sim >= max:
            max = sim
            question = each

    return question

# q = question_similarity('I am nervous', questions)
# indices = [i for i, x in enumerate(questions) if x == q]

# references = []
# for idx in indices:
#    references.append(answers[idx])
# print(references)

def get_blue_score(user_text, bot_answer, questions, answers):
    # get similar questions
    q = question_similarity(user_text, questions)

    # get question index for each matching question
    indices = [i for i, x in enumerate(questions) if x == q]

    # use index to retrieve answers
    references = []
    for idx in indices:
        references.append(answers[idx])

    bleu_score = sentence_bleu([x.split() for x in references], bot_answer, weights=(1, 0, 0, 0))
    return bleu_score


def get_meteor_score(user_text, bot_answer, questions, answers):
    q = question_similarity(user_text, questions)
    # print(q)
    indices = [i for i, x in enumerate(questions) if x == q]
    references = []
    for idx in indices:
        references.append(answers[idx])

    # print(references)
    score_list = []

    for each in references:
        each = each.split()
        # bot_answer = bot_answer.split()
        score = nltk.translate.meteor_score.meteor_score([each], bot_answer)
        score_list.append(score)

    # print('Meteor Score: ', max(score_list))
    # print(score_list)
    return max(score_list)
    # ['welcome user_firstname how are you today', 'hello user_firstname hows your day going', 'hey user_firstname how are you']



if __name__ == "__main__":
    df = pd.read_csv('data/evaluation_data.csv')
    df['questions'] = df['questions'].apply(text_cleaning)
    df['answers'] = df['answers'].apply(text_cleaning)


    questions = []
    answers = []
    for row in df.iterrows():
        questions.append(row[1]['questions'])
        answers.append(row[1]['answers'])

    #p = get_blue_score('hello', 'welcome user_firstname how are you today', questions, answers)
    #print(p)

    db_manager = DatabaseManager('EmpathBot.db')
    db_manager.check_database()
    qa_list = db_manager.fetch_question_answer()
    total_bleu_scores = 0
    total_meteor_scores = 0
    count = 0
    bleu_score_list = []
    meteor_score_list = []
    for each in qa_list:
        # clean user input
        user_input = each[0]
        user_input = text_cleaning(user_input)

        # clean bot responses
        bot_response = each[1]
        bot_response = text_cleaning(bot_response)

        bleu_score = get_blue_score(user_input, bot_response.split(), questions, answers)

        bleu_score_list.append(bleu_score)
        total_bleu_scores += bleu_score

        meteor_score = get_meteor_score(user_input, bot_response.split(), questions, answers)
        meteor_score_list.append(meteor_score)
        total_meteor_scores += meteor_score

        count += 1

    print('List of bleu scores: ', bleu_score_list)
    print('Average bleu score: ', total_bleu_scores/count)

    print('List of meteor scores: ', meteor_score_list)
    print('Average meteor score: ', total_meteor_scores/count)