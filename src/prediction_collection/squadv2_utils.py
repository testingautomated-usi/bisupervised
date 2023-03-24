import re
import string
from dataclasses import dataclass
from typing import List, Set

from datasets import load_dataset


def _normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace.
    
    Taken from https://rajpurkar.github.io/SQuAD-explorer/ (original eval script)"""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


@dataclass
class SquadV2TestSample:
    id: int
    title: str
    background: str
    practice_question: str
    practice_answer: str
    question: str
    answers: Set[str]

    @property
    def is_impossible(self) -> bool:
        return len(self.answers) == 0

    def is_exact_match(self, prediction: str) -> bool:
        normalized_gold_answers = [_normalize_answer(s) for s in self.answers]
        return _normalize_answer(prediction) in normalized_gold_answers


def get_squad_dataset() -> List[SquadV2TestSample]:
    dataset = load_dataset("squad_v2")['validation']
    test_set: List[SquadV2TestSample] = []

    entries_by_title = dict()
    for i, entry in enumerate(dataset):
        try:
            entries_by_title[entry['context']].append(entry)
        except KeyError:
            entries_by_title[entry['context']] = [entry]

    question_count = 0
    for question, entries in entries_by_title.items():
        practice_question = entries[0]['question']
        practice_answer = entries[0]['answers']['text'][0]

        for entry in entries[1:]:
            test_set.append(SquadV2TestSample(
                id=question_count,
                title=entry['title'],
                background=entry['context'],
                practice_question=practice_question,
                practice_answer=practice_answer,
                question=entry['question'],
                answers=set(entry['answers']['text']),
            ))
            question_count += 1

    return test_set


if __name__ == '__main__':
    get_squad_dataset()
