from nlputils.vocab_generator import VocabGenerator
from nlputils.tokenizer import BasicTokenizer

import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt='%Y-%m-%d  %H:%M:%S %a')


if __name__ == "__main__":
    tokenizer = BasicTokenizer(language='cn')
    gen = VocabGenerator(coverage=1.0)

    samples = ['《荒野大镖客：救赎2》拥有一个巨大的开放世界，而且充满活力，不过单人模式下在这个世界中逛久了，总是会感觉有些无聊。于是下面这位玩家Alex Tanaka决定让自己化身为西部大恶人。他的做法就是绑架游戏中每个郡的治安官，然后在风景宜人的地方与他们玩决斗游戏。决斗的结果他会直接远景截图，当成风景照传到网上，当然他基本只发自己吊打对方的照片。']
    seg_samples = [tokenizer.tokenize(x) for x in samples]
    print(seg_samples)
    gen.generate_vocab(seg_samples)
    vocab = gen.get_vocab()
    print(vocab[:20])
    # VOCAB_PATH = 'data/vocab.txt'
    tokenizer.load_vocab(vocab)
    # print(tokenizer.encode(samples[0], max_length=150, padding_mode='right'))