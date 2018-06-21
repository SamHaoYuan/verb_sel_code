# -*- coding: utf-8 -*-
import random
import json
import pandas as pd


class QuestionnaireSurvey:
    def __init__(self, questions_number, language='Eng'):
        self.questions_number = questions_number
        self.questions = []
        self.language = language

    def generate_questions(self, subs, pers, baseline_estimator, proposed_estimator, filename):
        print("generate_questions")
        question_list = []
        verbs_list = []
        i = 0
        while i < self.questions_number:
            if i % 10 == 0:
                question_list.clear()
            sub = random.choice(subs)
            per = random.choice(pers)
            print(i)
            if baseline_estimator.get_verbs(per):
                # baseline_verb = random.choice(baseline_estimator.get_verbs(per))
                baseline_verb = baseline_estimator.get_verbs(per)
            else:
                baseline_verb = random.choice(baseline_estimator.verbs)
            proposed_verb = proposed_estimator.p_verb(per)
            pairs = [per, baseline_verb, proposed_verb]
            if baseline_verb == proposed_verb:
                continue
            if pairs in question_list:
                continue
            question_list.append(pairs)
            verbs = {"baseline": baseline_verb, "proposed": proposed_verb}
            question = Question(sub, verbs, per, self.language)
            self.questions.append(question)
            verbs_list.append(pairs)
            i += 1
        verbs = pd.DataFrame(verbs_list, columns=['per', 'baseline_verb', 'proposed_verb'])
        verbs.to_csv(filename+'_verbs')
        print("generate finish")

    def generate_survey(self, filename):
        i = 0
        survey_file = open(filename+'.txt', 'a')
        open_file = open(filename, 'a')
        for question in self.questions:
            i += 1
            sentences = question.get_sentence()
            answer = question.get_answer()
            # options = question.get_option()
            survey_file.write("  "+str(i)+'.'+'\n')
            survey_file.write("  (1) "+sentences['a']+'\n')
            survey_file.write("  (2) " + sentences['b'] + '\n')
            if self.language == 'Eng':
                survey_file.write("  a. " + '(1) sounds better.' + '\n')
                survey_file.write("  b. " + '(2) sounds better.' + '\n')
                survey_file.write("  c. " + 'They are equally good.' + '\n')
                question_ins = {'q_id': i, 's_1': sentences['a'], 's_2': sentences['b'], 'option_a': '(1) sounds better',
                                'option_b': '(2) sounds better', 'option_c': 'They are equally good.', 'answer': answer}
            else:
                survey_file.write("  a. " + '(1)更好' + '\n')
                survey_file.write("  b. " + '(2)更好' + '\n')
                survey_file.write("  c. " + '一样好' + '\n')
                question_ins = {'q_id': i, 's_1': sentences['a'], 's_2': sentences['b'], 'option_a': '(1)更好',
                                'option_b': '(2)更好', 'option_c': '一样好', 'answer': answer}
            survey_file.write("  answer：" + answer + '\r\n')
            open_file.write(json.dumps(question_ins)+'\n')


class Question:
    def __init__(self, sub, verbs, per, language = 'Eng'):
        """

        :param sub:  句子主语（三选一）
        :param verbs: 动词选项（baseline动词与我们所提出的动词）
        :param per: 从数据集中得到的百分比
        """
        self.sub = sub
        self.per = per
        self.verbs = verbs
        # self.sentences = ""
        # self.option = {}
        options = ['a', 'b']
        self.answer = random.choice(options)
        self.language = language

    def get_sentence(self):
        sentence = {}
        verb_baseline = self.verbs['baseline']
        verb_proposed = self.verbs['proposed']
        options = ['a', 'b']
        self.answer = random.choice(options)
        if self.language == 'Eng':
            sentence[self.answer] = self.sub + " " + verb_proposed + " " + str(self.per) + "%."
        else:
            sentence[self.answer] = self.sub + verb_proposed + str(self.per) + "%."
        options.remove(self.answer)
        base_option = options.pop()
        if self.language == 'Eng':
            sentence[base_option] = self.sub + " " + verb_baseline + " " + str(self.per) + "%."
        else:
            sentence[base_option] = self.sub + verb_baseline + str(self.per) + "%."
        return sentence

    def get_option(self):
        options = {'a': "句子（1）比句子（2）表述得更加自然", 'b': "句子（2）比句子（1）表述得更加自然"}
        return options

    def get_answer(self):
        return self.answer

