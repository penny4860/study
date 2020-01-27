
from gensim.models import Word2Vec

sentences = [
                ["헤링본", "비치원피스", "여름", "롱", "미니원피스", "휴양지원피스", "바캉스룩", "비치웨어", "데일리", "나시"],
                ["플랜비", "최대30%쿠폰", "시즌오프", "봄신상", "니트", "가디건", "후드티", "맨투맨", "원피스", "스커트", "팬츠", "등"],
                ['it',   'is', 'a',   'bad',       'product'],
                ['that', 'is', 'the', 'worst',     'product']
            ]

# 문장을 이용하여 단어와 벡터를 생성한다.
model = Word2Vec(sentences, size=300, window=3, min_count=1, workers=1)

# 단어벡터를 구한다.
word_vectors = model.wv

vocabs = word_vectors.vocab.keys()
word_vectors_list = [word_vectors[v] for v in vocabs]

print(vocabs)
print("여름", model.most_similar(positive=["여름"], topn=100))
