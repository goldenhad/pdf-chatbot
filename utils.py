class QAEntry:
    def __init__(self, question, answer, yes_votes=0, no_votes=0):
        self.question = question
        self.answer = answer
        self.yes_votes = yes_votes
        self.no_votes = no_votes

    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "yes_votes": self.yes_votes,
            "no_votes": self.no_votes,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data["question"], data["answer"], data["yes_votes"], data["no_votes"])